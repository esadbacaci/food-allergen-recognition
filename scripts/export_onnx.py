# -*- coding: utf-8 -*-
"""Export the deployed model to ONNX and verify the conversion on the whole test set.

    python scripts/export_onnx.py [--app ../10_web_uygulama]

Deployed model = the seed of the `final` group with the highest VALIDATION accuracy (the test
set plays no part in the choice). Temperature and abstention threshold are taken from
analysis/calibration_final.json (both fitted on validation logits by scripts/evaluate.py).

Verification (conversion only, not the application's image decoding): PyTorch FP32 on the GPU
versus ONNX Runtime FP32 on the CPU (the production runtime) on identical preprocessed test
tensors; logit and probability differences, top-1 agreement and both accuracies.
Outputs: 03_deneyler/final/<model>/final_model.onnx, analysis/onnx_verification.json,
         data/deploy_config.json (copied to <app>/ml/ with the model when --app is given).
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.data import gpu_preprocess, make_loaders  # noqa: E402
from foodallergen.models import build_model  # noqa: E402

OPSET = 17


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default=None, help="web application folder (receives ml/final_model.onnx)")
    args = ap.parse_args()

    seeds = sorted((paths.RUNS / "final").glob("*/seed*"))
    val = {rd: json.loads((rd / "metrics.json").read_text())["val"]["acc"] for rd in seeds if (rd / "DONE").exists()}
    rd = max(val, key=val.get)
    cfg = json.loads((rd / "config.json").read_text())
    calib = json.loads((paths.RUNS / "analysis" / "calibration_final.json").read_text())[f"{rd.parent.name}/{rd.name}"]
    model = build_model(cfg["model"], 206, pretrained=False)
    model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu"))
    model.eval()
    size = cfg["img_size"]

    onnx_path = rd.parent / "final_model.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, size, size), str(onnx_path), opset_version=OPSET,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    print(f"exported {rd.parent.name}/{rd.name} -> {onnx_path} ({onnx_path.stat().st_size / 2**20:.1f} MB)", flush=True)

    import onnxruntime as ort
    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    data = make_loaders(64, 1, size, "strong", num_workers=0, eval_workers=4, eval_batch_size=64,
                        image_dir=cfg["image_dir"])
    data["train"] = None
    gpu = model.cuda()
    pt, ox, ys = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(data["test"]):
            xc = gpu_preprocess(x, torch.device("cpu"))
            pt.append(gpu(xc.cuda()).float().cpu().numpy())
            ox.append(sess.run(None, {"input": xc.numpy()})[0])
            ys.append(y.numpy())
            if i % 50 == 0:
                print(f"batch {i}/{len(data['test'])} {time.time() - t0:.0f}s", flush=True)
    pt, ox, y = np.concatenate(pt), np.concatenate(ox), np.concatenate(ys)
    dp = np.abs(softmax(pt) - softmax(ox))
    res = dict(run=f"{rd.parent.name}/{rd.name}", validation_acc=val[rd], opset=OPSET, img_size=size,
               onnx_mb=round(onnx_path.stat().st_size / 2**20, 1), n_test=int(len(y)),
               logit_mean_abs_diff=float(np.abs(pt - ox).mean()), logit_max_abs_diff=float(np.abs(pt - ox).max()),
               prob_mean_abs_diff=float(dp.mean()), prob_max_abs_diff=float(dp.max()),
               top1_agreement=float((pt.argmax(1) == ox.argmax(1)).mean()),
               top1_disagreements=int((pt.argmax(1) != ox.argmax(1)).sum()),
               acc_pytorch_fp32=float((pt.argmax(1) == y).mean()), acc_onnx_cpu=float((ox.argmax(1) == y).mean()))
    (paths.RUNS / "analysis" / "onnx_verification.json").write_text(json.dumps(res, indent=2))
    deploy = dict(model=cfg["model"], run=res["run"], input_size=size, resize_shorter_side=int(round(size / 0.875)),
                  interpolation="bicubic", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                  temperature=calib["temperature"], abstain_threshold=calib["val_threshold"]["threshold"],
                  abstain_rule="temperature-scaled max probability below the threshold that gives 95% "
                               "selective accuracy on the validation set -> ask the user to confirm among top-3",
                  onnx_sha256=__import__("hashlib").sha256(onnx_path.read_bytes()).hexdigest())
    (paths.REPO_DATA / "deploy_config.json").write_text(json.dumps(deploy, indent=2))
    print(json.dumps(res, indent=2))
    if args.app:
        ml = Path(args.app) / "ml"
        shutil.copy2(onnx_path, ml / "final_model.onnx")
        shutil.copy2(paths.REPO_DATA / "deploy_config.json", ml / "deploy_config.json")
        print("copied to", ml)


if __name__ == "__main__":
    main()
