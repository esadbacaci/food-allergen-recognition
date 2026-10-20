# -*- coding: utf-8 -*-
"""Robustness of the Turkish test accuracy to near-duplicates looser than the deduplication threshold.

    python scripts/check_near_duplicates.py

For every Turkish test image, looks for a training image of the same class whose 64-bit perceptual hash differs
in at most 8 or 10 bits (deduplication used 5), and reports the MaxViT-T accuracy on the Turkish test images
with and without the flagged images (groups main and final, three seeds).
Output: 03_deneyler/analysis/turkish_near_duplicate_check.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.data import read_class_map  # noqa: E402
from foodallergen.experiments import SEEDS  # noqa: E402
from foodallergen.hashing import popcount64  # noqa: E402

THRESHOLDS = (8, 10)


def main():
    rows = list(csv.DictReader(open(paths.MANIFEST_V2, encoding="utf-8")))
    idx = {c["class_id"]: i for i, c in enumerate(read_class_map())}
    test = [r for r in rows if r["split"] == "test"]
    train = [r for r in rows if r["split"] == "train"]
    y = np.array([idx[r["cls"]] for r in test])
    h_train = np.array([int(r["phash"], 16) for r in train], dtype=np.uint64)
    c_train = np.array([idx[r["cls"]] for r in train])
    is_tr = np.array([r["source"] != "food101" for r in test])
    flags = {}
    for t in THRESHOLDS:
        f = np.zeros(len(test), bool)
        for i in np.where(is_tr)[0]:
            d = popcount64(h_train ^ np.uint64(int(test[i]["phash"], 16)))
            f[i] = ((d <= t) & (c_train == y[i])).any()
        flags[t] = f
    out = dict(n_turkish_test=int(is_tr.sum()), n_flagged={str(t): int(f.sum()) for t, f in flags.items()})
    for grp in ("main", "final"):
        acc = {0: [], **{t: [] for t in THRESHOLDS}}
        for s in SEEDS:
            z = np.load(paths.RUNS / grp / "maxvit_t" / f"seed{s}" / "logits_test.npz")
            assert (z["labels"] == y).all()
            ok = z["logits"].argmax(1) == y
            acc[0].append(ok[is_tr].mean())
            for t in THRESHOLDS:
                acc[t].append(ok[is_tr & ~flags[t]].mean())
        out[grp] = {str(t): [float(np.mean(v)) * 100, float(np.std(v, ddof=1)) * 100] for t, v in acc.items()}
    (paths.RUNS / "analysis" / "turkish_near_duplicate_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
