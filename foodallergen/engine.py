"""Training / evaluation loop shared by every experiment.

One run = one (model, recipe, seed). Outputs in <run_dir>:
  config.json, env.json, train.log, history.csv,
  best.pt (weights of the epoch with the highest validation accuracy),
  last.pt (resume state, deleted when the run finishes),
  logits_val.npz / logits_test.npz (float16 logits + labels), metrics.json, DONE
"""
import csv
import gc
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .data import gpu_preprocess, make_loaders
from .models import TRANSFORMERS, build_model, param_groups

DEFAULTS = dict(
    model="effb0", seed=1, epochs=20, warmup_epochs=2, lr=1e-3, weight_decay=0.05,
    optimizer="adamw", schedule="cosine", label_smoothing=0.1, class_weights=True,
    llrd=None, aug="strong", img_size=224, effective_batch=64, physical_batch=64,
    num_workers=8, eval_workers=4, eval_test=True, keep_best_weights=True, image_dir="images",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_info():
    return dict(python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                platform=platform.platform())


def lr_factor(step, total, warmup, schedule):
    if schedule == "constant":
        return 1.0
    if step < warmup:
        return 1e-3 + (1 - 1e-3) * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def predict(model, loader, device, channels_last):
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        x = gpu_preprocess(x, device, channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        outs.append(out.float().cpu())
        ys.append(y)
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def summarize(logits, labels):
    pred = logits.argmax(1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    return dict(acc=float((pred == labels).mean()),
                top5=float((top5 == labels[:, None]).any(1).mean()),
                macro_f1=float(f1_score(labels, pred, average="macro", zero_division=0)))


def run(cfg, run_dir):
    cfg = {**DEFAULTS, **cfg}
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE").exists():
        print(f"[skip] {run_dir.name} already finished")
        return json.loads((run_dir / "metrics.json").read_text())
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (run_dir / "env.json").write_text(json.dumps(env_info(), indent=2))
    log_f = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    accum = cfg["effective_batch"] // cfg["physical_batch"]
    assert accum * cfg["physical_batch"] == cfg["effective_batch"]
    data = make_loaders(cfg["physical_batch"], cfg["seed"], cfg["img_size"], cfg["aug"],
                        cfg["num_workers"], cfg["eval_workers"], image_dir=cfg["image_dir"])
    n_cls = len(data["class_ids"])
    model = build_model(cfg["model"], n_cls).to(device)
    channels_last = cfg["model"] not in TRANSFORMERS
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    weight = data["class_weights"].to(device) if cfg["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["label_smoothing"])
    if cfg["optimizer"] == "adamw":
        groups = param_groups(model, cfg["model"], cfg["lr"], cfg["weight_decay"], cfg["llrd"])
        opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    elif cfg["optimizer"] == "adam":  # v1-style: plain Adam, no weight decay
        groups = param_groups(model, cfg["model"], cfg["lr"], 0.0, cfg["llrd"])
        opt = torch.optim.Adam(groups, lr=cfg["lr"])
    else:
        raise ValueError(cfg["optimizer"])

    steps_per_epoch = len(data["train"]) // accum
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = steps_per_epoch * cfg["warmup_epochs"] if cfg["schedule"] != "constant" else 0

    start_epoch, step, best_acc, best_epoch, best_state, history = 0, 0, -1.0, -1, None, []
    if (run_dir / "last.pt").exists():
        st = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch, step = st["epoch"] + 1, st["step"]
        best_acc, best_epoch, history = st["best_acc"], st["best_epoch"], st["history"]
        best_state = st["best_state"]
        torch.set_rng_state(st["torch_rng"])
        log(f"resumed after epoch {st['epoch'] + 1}")

    log(f"run {run_dir.name} | {cfg} | train/val/test = {len(data['train'].dataset)}/"
        f"{len(data['val'].dataset)}/{len(data['test'].dataset)} | classes {n_cls} | accum {accum}")
    torch.cuda.reset_peak_memory_stats()
    t_train = 0.0
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t0 = time.time()
        # running sums stay on the GPU: one synchronisation per epoch instead of per step
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        seen = 0
        opt.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(data["train"]):
            x = gpu_preprocess(x, device, channels_last, erase=cfg["aug"] == "strong")
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            loss = criterion(out.float(), y)
            (loss / accum).backward()
            if (it + 1) % accum == 0:
                f = lr_factor(step, total_steps, warmup_steps, cfg["schedule"])
                for g in opt.param_groups:
                    g["lr"] = cfg["lr"] * g.get("lr_scale", 1.0) * f
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            loss_sum += loss.detach() * y.size(0)
            correct += (out.argmax(1) == y).sum()
            seen += y.size(0)
        loss_sum, correct = loss_sum.item(), correct.item()
        train_time = time.time() - t0
        t_train += train_time

        val_logits, val_y = predict(model, data["val"], device, channels_last)
        vm = summarize(val_logits, val_y)
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(val_logits),
                                                     torch.from_numpy(val_y)))
        rec = dict(epoch=epoch + 1, train_loss=loss_sum / seen, train_acc=correct / seen,
                   val_loss=val_loss, val_acc=vm["acc"], val_top5=vm["top5"],
                   val_macro_f1=vm["macro_f1"], lr=opt.param_groups[0]["lr"],
                   epoch_seconds=round(train_time, 1))
        history.append(rec)
        improved = vm["acc"] > best_acc
        if improved:
            best_acc, best_epoch = vm["acc"], epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"epoch {epoch + 1}/{cfg['epochs']} {train_time:.0f}s | train loss {rec['train_loss']:.4f} "
            f"acc {rec['train_acc']:.4f} | val loss {val_loss:.4f} acc {vm['acc']:.4f} "
            f"top5 {vm['top5']:.4f} f1 {vm['macro_f1']:.4f}{' *' if improved else ''}")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch, step=step,
                        best_acc=best_acc, best_epoch=best_epoch, history=history,
                        best_state=best_state, torch_rng=torch.get_rng_state()),
                   run_dir / "last.pt")

    # release the 8 persistent training workers before the final val/test passes (memory)
    data["train"] = None
    gc.collect()

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    model.load_state_dict(best_state)
    val_logits, val_y = predict(model, data["val"], device, channels_last)
    np.savez_compressed(run_dir / "logits_val.npz", logits=val_logits.astype(np.float16), labels=val_y)
    metrics = dict(run=run_dir.name, best_epoch=best_epoch, val=summarize(val_logits, val_y),
                   peak_gpu_mem_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                   mean_epoch_seconds=round(float(np.mean([h["epoch_seconds"] for h in history])), 1))
    if cfg["eval_test"]:
        test_logits, test_y = predict(model, data["test"], device, channels_last)
        np.savez_compressed(run_dir / "logits_test.npz", logits=test_logits.astype(np.float16),
                            labels=test_y)
        metrics["test"] = summarize(test_logits, test_y)
    if cfg["keep_best_weights"]:
        torch.save(best_state, run_dir / "best.pt")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "last.pt").unlink(missing_ok=True)
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"finished: best epoch {best_epoch} | {json.dumps(metrics)}")
    log_f.close()
    return metrics
