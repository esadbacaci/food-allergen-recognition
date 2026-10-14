#!/bin/bash
# Unattended experiment chain (about 3.5 days on one RTX 5060). Safe to restart: finished
# runs are skipped and interrupted runs resume from their last completed epoch.
#   bash scripts/run_all.sh >> ../03_deneyler/run_all.log 2>&1
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
stamp() { echo "=== $(date '+%Y-%m-%d %H:%M:%S') $*"; }

stamp "1/8 learning-rate sweep";   python scripts/run.py sweep
stamp "2/8 select learning rates"; python scripts/select_lr.py || { stamp "SELECT_LR_FAILED"; exit 1; }
stamp "3/8 controlled comparison"; python scripts/run.py main
stamp "4/8 select final backbone"; python scripts/select_final.py || { stamp "SELECT_FINAL_FAILED"; exit 1; }
stamp "5/8 final model (320 px)";  python scripts/run.py final
stamp "6/8 recipe ablation";       python scripts/run.py ablation
stamp "7/8 evaluation"
python scripts/evaluate.py
for m in resnet18 effb0 effv2s convnext_t swin_t vit_b16; do
  python scripts/evaluate_allergen.py --model "$m"
done
python scripts/evaluate_allergen.py --group final --model "$(python -c "import json;print(json.load(open('data/final_choice.json'))['model'])")"
stamp "8/8 efficiency";            python scripts/benchmark_inference.py
stamp "ALL_EXPERIMENTS_DONE"
