"""Pick, for every backbone, the grid learning rate with the best validation accuracy.

    python scripts/select_lr.py      -> data/selected_lr.json + data/lr_sweep.csv
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.experiments import LR_GRID, MODELS, SELECTED_LR_FILE  # noqa: E402

rows, best = [], {}
for m in MODELS:
    for lr in LR_GRID:
        f = paths.RUNS / "sweep" / m / f"lr{lr:g}" / "metrics.json"
        if not f.exists():
            sys.exit(f"missing sweep result: {f}")
        acc = json.loads(f.read_text())["val"]["acc"]
        rows.append(dict(model=m, lr=lr, val_acc=round(acc, 4)))
        if m not in best or acc > best[m][1]:
            best[m] = (lr, acc)
    edge = best[m][0] in (LR_GRID[0], LR_GRID[-1])
    print(f"{m:12s} best lr {best[m][0]:g} (val acc {best[m][1]:.4f}){'  <- grid edge' if edge else ''}")

SELECTED_LR_FILE.write_text(json.dumps({m: lr for m, (lr, _) in best.items()}, indent=2))
with open(paths.REPO_DATA / "lr_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "lr", "val_acc"])
    w.writeheader()
    w.writerows(rows)
