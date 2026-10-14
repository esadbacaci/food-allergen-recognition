"""Training-step speed and peak memory per backbone (synthetic input, bf16 autocast).

Used once to choose the physical batch size per model (effective batch is always 64)
and to estimate the GPU budget.   python scripts/benchmark_gpu.py
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen.models import SPECS, TRANSFORMERS, build_model  # noqa: E402


def bench(name, batch, n_cls=207, iters=20):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(name, n_cls, pretrained=False).cuda()
    cl = name not in TRANSFORMERS
    if cl:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    x = torch.randn(batch, 3, 224, 224, device="cuda")
    if cl:
        x = x.contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, n_cls, (batch,), device="cuda")
    times = []
    for i in range(iters + 5):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        loss = crit(out.float(), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 5:
            times.append(time.perf_counter() - t)
    mem = torch.cuda.max_memory_allocated() / 2**20
    del model, opt, x
    return sum(times) / len(times), mem


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    res = {}
    for name in SPECS:
        for batch in (64, 32, 16):
            try:
                t, mem = bench(name, batch)
                ips = batch / t
                res[name] = dict(physical_batch=batch, step_s=round(t, 4), img_per_s=round(ips, 1),
                                 peak_mem_mb=round(mem), epoch_min_per_100k=round(1e5 / ips / 60, 1))
                print(name, res[name], flush=True)
                break
            except torch.cuda.OutOfMemoryError:
                print(name, batch, "OOM", flush=True)
                torch.cuda.empty_cache()
    out = Path(__file__).resolve().parents[1] / "data" / "gpu_benchmark.json"
    out.write_text(json.dumps(res, indent=2))
