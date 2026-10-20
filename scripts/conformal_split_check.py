# -*- coding: utf-8 -*-
"""Does the risk-aware rule keep its sensitivity when calibration data are not reused?

    python scripts/conformal_split_check.py

In the main analysis one validation set selects the checkpoint, fits the temperature and sets the conformal
allergen thresholds. Here the validation images are split at random (stratified by class, 10 repetitions) into
two disjoint halves: the temperature is fitted on half A and the thresholds on half B only, and the rule is
evaluated on the test set. Deployed MaxViT-T, three seeds, database v2.1.
Output: 03_deneyler/analysis/conformal_split_check.json
"""
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

REPS = 10


def main():
    ids = [c["class_id"] for c in read_class_map()]
    m, _, names = E.profile_matrix(paths.REPO_DATA / "allergen_db_v2", ids)
    mf = m.astype(float)
    g = names.index("Cereals containing gluten")
    out = []
    for s in SEEDS:
        rd = paths.RUNS / "final" / "maxvit_t" / f"seed{s}"
        zv, zt = np.load(rd / "logits_val.npz"), np.load(rd / "logits_test.npz")
        vl, vy, tl, ty = zv["logits"].astype(np.float32), zv["labels"], zt["logits"].astype(np.float32), zt["labels"]
        truth = m[ty]
        for rep in range(REPS):
            rng = np.random.default_rng(1000 * s + rep)
            a = np.zeros(len(vy), bool)
            for c in np.unique(vy):
                idx = np.where(vy == c)[0]
                a[rng.choice(idx, len(idx) // 2, replace=False)] = True
            T = M.fit_temperature(vl[a], vy[a])
            rb = M.calibrated_probs(vl[~a], T) @ mf
            theta = np.array([E.threshold_for_sensitivity(rb[:, j], m[vy[~a], j], 0.99) if m[vy[~a], j].any()
                              else np.inf for j in range(m.shape[1])])
            warn = (M.calibrated_probs(tl, T) @ mf) >= theta
            miss = truth & ~warn
            fnr = [miss[:, j].sum() / truth[:, j].sum() for j in range(m.shape[1]) if truth[:, j].any()]
            out.append(dict(seed=s, rep=rep, max_fnr=float(max(fnr)), gluten_fnr=float(fnr[g]),
                            images_missed=float(miss.any(1).mean()),
                            all_allergens_meet_target=bool(max(fnr) <= 0.01)))
    summ = {k: [float(np.mean([o[k] for o in out])), float(np.max([o[k] for o in out]))]
            for k in ("max_fnr", "gluten_fnr", "images_missed")}
    summ["share_runs_all_allergens_meet_target"] = float(np.mean([o["all_allergens_meet_target"] for o in out]))
    summ["n_runs"] = len(out)
    (paths.RUNS / "analysis" / "conformal_split_check.json").write_text(json.dumps(dict(summary=summ, runs=out),
                                                                                  indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
