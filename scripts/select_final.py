"""Choose the backbone for the deployed model from the VALIDATION accuracy of `main`.

    python scripts/select_final.py   -> data/final_choice.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.experiments import FINAL_CHOICE_FILE, MODELS, SEEDS, selected_lr  # noqa: E402

val = {}
for m in MODELS:
    accs = []
    for s in SEEDS:
        f = paths.RUNS / "main" / m / f"seed{s}" / "metrics.json"
        if f.exists():
            accs.append(json.loads(f.read_text())["val"]["acc"])
    if len(accs) == len(SEEDS):
        val[m] = accs
if not val:
    sys.exit("no complete main results")
best = max(val, key=lambda m: np.mean(val[m]))
choice = dict(model=best, lr=selected_lr()[best], rule="highest mean validation top-1 over seeds",
              validation_accuracy={m: [round(a, 4) for a in v] for m, v in val.items()})
FINAL_CHOICE_FILE.write_text(json.dumps(choice, indent=2))
print(json.dumps(choice, indent=2))
