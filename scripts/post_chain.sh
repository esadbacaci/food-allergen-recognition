#!/bin/bash
# Steps added after run_all.sh had started (run_all.sh must not be edited while it runs).
# Run once run_all.log shows ALL_EXPERIMENTS_DONE.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
stamp() { echo "=== $(date "+%Y-%m-%d %H:%M:%S") $*"; }
stamp "post 1/3 OOD: deployed food gate";      python scripts/evaluate_ood.py --part gate
stamp "post 2/3 OOD: classifier (main)";       python scripts/evaluate_ood.py --part classifier --group main
stamp "post 3/3 OOD: classifier (final)";      python scripts/evaluate_ood.py --part classifier --group final
stamp "POST_CHAIN_DONE"
