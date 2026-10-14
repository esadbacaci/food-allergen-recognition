#!/bin/bash
# Unattended experiment chain (~2.5 days on one RTX 5060). Safe to restart: finished runs are
# skipped and interrupted runs resume from their last epoch.
#   bash scripts/run_all.sh  >> ../03_deneyler/run_all.log 2>&1
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
stamp() { echo "=== $(date '+%Y-%m-%d %H:%M:%S') $*"; }

stamp "sweep start";     python scripts/run.py sweep
stamp "select lr";       python scripts/select_lr.py || { stamp "SELECT_LR_FAILED"; exit 1; }
stamp "main start";      python scripts/run.py main
stamp "ablation start";  python scripts/run.py ablation
stamp "evaluate";        python scripts/evaluate.py
stamp "allergen eval"
for m in resnet18 effb0 effv2s convnext_t swin_t vit_b16; do
  python scripts/evaluate_allergen.py --model "$m"
done
stamp "efficiency";      python scripts/benchmark_inference.py
stamp "ALL_EXPERIMENTS_DONE"
