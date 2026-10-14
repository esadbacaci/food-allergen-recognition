# -*- coding: utf-8 -*-
"""Progress report of the experiment chain (read-only; safe to run at any time).

    python scripts/status.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.experiments import (ABLATIONS, FINAL, LR_GRID, MODELS, SEEDS,  # noqa: E402
                                      SWEEP_EPOCHS)

# planned runs per group: (run_dir, model, epochs, relative cost per epoch)
FINAL_COST = (FINAL["img_size"] / 224) ** 2


def planned():
    plan = []
    for m in MODELS:
        for lr in LR_GRID:
            plan.append(("sweep", paths.RUNS / "sweep" / m / f"lr{lr:g}", m, SWEEP_EPOCHS, 1.0))
    for s in SEEDS:
        for m in MODELS:
            plan.append(("main", paths.RUNS / "main" / m / f"seed{s}", m, 20, 1.0))
    choice = paths.REPO_DATA / "final_choice.json"
    fm = json.loads(choice.read_text())["model"] if choice.exists() else "effv2s"
    for s in SEEDS:
        plan.append(("final", paths.RUNS / "final" / fm / f"seed{s}", fm, FINAL["epochs"], FINAL_COST))
    for s in SEEDS:
        for a in ABLATIONS:
            plan.append(("ablation", paths.RUNS / "ablation" / a / f"seed{s}", "effb0", 20, 1.0))
    return plan


def epochs_done(run_dir):
    log = run_dir / "train.log"
    if not log.exists():
        return 0, []
    secs = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if " epoch " in line and "s | train loss" in line:
            secs.append(float(line.split(" epoch ")[1].split()[1].rstrip("s")))
    return len(secs), secs


def main():
    plan = planned()
    per_model = defaultdict(list)
    for _, rd, m, _, cost in plan:
        _, secs = epochs_done(rd)
        per_model[m] += [s / cost for s in secs[1:]]  # skip first epoch (worker start-up)
    # fallback seconds/epoch at 224 px when a model has not run yet (measured benchmarks)
    fallback = {"resnet18": 80, "effb0": 140, "effv2s": 300, "convnext_t": 270, "swin_t": 360, "vit_b16": 600}
    sec_per_epoch = {m: (sum(v) / len(v) if v else fallback[m]) + 40 for m, v in
                     ((m, per_model[m]) for m in fallback)}  # +40 s validation

    groups = defaultdict(lambda: [0, 0])
    remaining, current = 0.0, None
    for g, rd, m, ep, cost in plan:
        groups[g][1] += 1
        if (rd / "DONE").exists():
            groups[g][0] += 1
            continue
        n, _ = epochs_done(rd)
        if n and current is None:
            current = (g, rd, n, ep)
        remaining += (ep - n) * sec_per_epoch[m] * cost
    print(f"Durum ({time.strftime('%Y-%m-%d %H:%M')})")
    for g in ("sweep", "main", "final", "ablation"):
        done, total = groups[g]
        print(f"  {g:9s} {done:3d}/{total:3d} koşu tamamlandı")
    if current:
        g, rd, n, ep = current
        print(f"  şu an: {rd.relative_to(paths.RUNS)}  (epoch {n}/{ep})")
    log = paths.RUNS / "run_all_stderr.log"
    if log.exists() and "Traceback" in log.read_text(encoding="utf-8", errors="ignore"):
        print("  UYARI: run_all_stderr.log içinde hata kaydı var, kontrol edin.")
    h = remaining / 3600
    print(f"  tahmini kalan süre: {h:.1f} saat (~{time.strftime('%d.%m %H:%M', time.localtime(time.time() + remaining))})")


if __name__ == "__main__":
    main()
