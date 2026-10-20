# -*- coding: utf-8 -*-
"""Allergen metrics of the deployed pipeline with the stricter reference used in the dietitian audit.

    python scripts/allergen_strict_reference.py

The policies are unchanged (an allergen is reported if the profile lists it at any presence level; risk thresholds
fitted on the validation set as in evaluate_allergen_risk.py). Only the reference changes: a test image counts as
positive for an allergen only if its true dish lists it as 'always' or 'often' (Her zaman / Sıklıkla), the
definition under which the database was audited. Deployed MaxViT-T, three seeds, database v2.1.
Output: 03_deneyler/analysis/allergen_strict_reference.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from foodallergen import metrics as M  # noqa: E402
from foodallergen import paths  # noqa: E402
from foodallergen.data import read_class_map  # noqa: E402
from foodallergen.experiments import SEEDS  # noqa: E402
import evaluate_allergen_risk as E  # noqa: E402

DB = paths.REPO_DATA / "allergen_db_v2"
STRICT = {"Her zaman", "Sıklıkla"}


def strict_matrix(class_ids):
    idx = {c: i for i, c in enumerate(class_ids)}
    m = np.zeros((len(class_ids), len(E.REGULATED)), bool)
    for r in csv.DictReader(open(DB / "food_allergens.csv", encoding="utf-8")):
        a = int(r["AlerjenID"])
        if a in E.REGULATED and r["class_id"] in idx and r["Miktar"] in STRICT:
            m[idx[r["class_id"]], E.REGULATED.index(a)] = True
    return m


def main():
    ids = [c["class_id"] for c in read_class_map()]
    m, covered, names = E.profile_matrix(DB, ids)
    ms = strict_matrix(ids)
    g = names.index("Cereals containing gluten") if "Cereals containing gluten" in names else 0
    per = []
    for s in SEEDS:
        rd = paths.RUNS / "final" / "maxvit_t" / f"seed{s}"
        zv, zt = np.load(rd / "logits_val.npz"), np.load(rd / "logits_test.npz")
        vl, vy, tl, ty = zv["logits"].astype(np.float32), zv["labels"], zt["logits"].astype(np.float32), zt["labels"]
        T = M.fit_temperature(vl, vy)
        pv, pt = M.calibrated_probs(vl, T), M.calibrated_probs(tl, T)
        rv, rt = pv @ m.astype(float), pt @ m.astype(float)
        theta = np.array([E.threshold_for_sensitivity(rv[:, j], m[vy, j], 0.99) if m[vy, j].any() else np.inf
                          for j in range(m.shape[1])])
        pred = pt.argmax(1)
        truth = ms[ty]
        res = {}
        for name, warn in (("top1", m[pred]), ("risk", rt >= theta)):
            miss = truth & ~warn
            res[name] = dict(images_missed=float(miss.any(1).mean()),
                             fnr_pooled=float(miss.sum() / truth.sum()),
                             gluten_fnr=float(miss[:, g].sum() / truth[:, g].sum()),
                             max_fnr=float(max(miss[:, j].sum() / truth[:, j].sum()
                                               for j in range(truth.shape[1]) if truth[:, j].any())))
        res["positives"] = int(truth.sum())
        per.append(res)
    out = {k: {q: [float(np.mean([p[k][q] for p in per])), float(np.std([p[k][q] for p in per], ddof=1))]
               for q in per[0][k]} for k in ("top1", "risk")}
    out["positives_mean"] = float(np.mean([p["positives"] for p in per]))
    out["positives_any_level"] = int(m[ty].sum())
    (paths.RUNS / "analysis" / "allergen_strict_reference.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
