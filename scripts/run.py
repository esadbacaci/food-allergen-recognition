"""Run one or more experiment groups; finished runs are skipped, interrupted runs resume.

    python scripts/run.py sweep
    python scripts/run.py main ablation
    python scripts/run.py main --only vit_b16
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import engine  # noqa: E402
from foodallergen.experiments import EXPERIMENTS  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", choices=list(EXPERIMENTS))
    ap.add_argument("--only", default=None, help="restrict to run dirs containing this text")
    args = ap.parse_args()
    for group in args.groups:
        for run_dir, cfg in EXPERIMENTS[group]():
            if args.only and args.only not in str(run_dir):
                continue
            t = time.time()
            try:
                engine.run(cfg, run_dir)
            except Exception:
                traceback.print_exc()
                print(f"[FAILED] {run_dir}", flush=True)
                continue
            print(f"[{group}] {run_dir} done in {(time.time() - t) / 3600:.2f} h", flush=True)
