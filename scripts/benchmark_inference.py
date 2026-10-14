"""Parameters, FLOPs and single-image latency under identical conditions for every backbone.

    python scripts/benchmark_inference.py [--weights main]   -> 03_deneyler/analysis/efficiency.csv

Conditions (the same for all models; answers reviewer 2/3 on Table 4):
  * batch size 1, 224x224, FP32, weights of a trained run when available;
  * GPU: CUDA events after torch.cuda.synchronize(), 50 warm-up + 300 timed runs, median;
  * CPU: the deployed path, ONNX Runtime CPUExecutionProvider (opset 17), same repeats,
    intra-op threads = physical cores; reported together with the CPU model;
  * FLOPs: torch.utils.flop_counter (counts matmuls inside attention), reported as GFLOPs
    = 2 x multiply-accumulates.
"""
import argparse
import csv
import io
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.data import read_class_map  # noqa: E402
from foodallergen.experiments import MODELS  # noqa: E402
from foodallergen.models import DISPLAY, build_model  # noqa: E402

WARMUP, REPEATS = 50, 300


def cpu_name():
    if platform.system() == "Windows":
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True).stdout.strip()
        return out or platform.processor()
    return platform.processor()


def gpu_latency(model, x):
    model = model.cuda().eval()
    x = x.cuda()
    times = []
    with torch.no_grad():
        for i in range(WARMUP + REPEATS):
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            model(x)
            e.record()
            torch.cuda.synchronize()
            if i >= WARMUP:
                times.append(s.elapsed_time(e))
    model.cpu()
    return statistics.median(times)


def onnx_cpu_latency(model, x):
    import onnxruntime as ort
    buf = io.BytesIO()
    torch.onnx.export(model.eval(), x, buf, opset_version=17, input_names=["input"],
                      output_names=["logits"], dynamo=False)
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() // 2 or 1
    sess = ort.InferenceSession(buf.getvalue(), so, providers=["CPUExecutionProvider"])
    feed = {"input": x.numpy()}
    times = []
    for i in range(WARMUP + REPEATS):
        t = time.perf_counter()
        sess.run(None, feed)
        if i >= WARMUP:
            times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times), so.intra_op_num_threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="main", help="experiment group whose seed1 weights to load")
    args = ap.parse_args()
    n_cls = len(read_class_map())
    x = torch.randn(1, 3, 224, 224)
    rows = []
    for m in MODELS:
        model = build_model(m, n_cls, pretrained=False)
        w = paths.RUNS / args.weights / m / "seed1" / "best.pt"
        if w.exists():
            model.load_state_dict(torch.load(w, map_location="cpu"))
        model.eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(x)
        gflops = fc.get_total_flops() / 1e9
        g = gpu_latency(model, x)
        c, threads = onnx_cpu_latency(model, x)
        rows.append({"Model": DISPLAY[m], "Parameters (M)": f"{params:.2f}",
                     "GFLOPs": f"{gflops:.2f}", "GPU latency (ms, median)": f"{g:.2f}",
                     "CPU ONNX latency (ms, median)": f"{c:.1f}", "CPU threads": threads})
        print(rows[-1], flush=True)
    out = paths.RUNS / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "efficiency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "efficiency_hardware.txt").write_text(
        f"GPU: {torch.cuda.get_device_name(0)}\nCPU: {cpu_name()}\ntorch {torch.__version__}\n"
        f"batch 1, 224x224, FP32, warm-up {WARMUP}, timed {REPEATS}, median\n", encoding="utf-8")


if __name__ == "__main__":
    main()
