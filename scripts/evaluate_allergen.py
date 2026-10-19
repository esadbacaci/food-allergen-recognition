# -*- coding: utf-8 -*-
"""Allergen-level safety metrics of the whole pipeline (classifier + database lookup).

    python scripts/evaluate_allergen.py [--db data/allergen_db_v1] [--model effv2s]

For every regulated allergen a (EU 1169/2011 Annex II = TGK Annex 1) and each test image:
  the image "contains a" if the database profile of its TRUE class lists a;
  the system "warns a" if the profile of the PREDICTED class lists a.
Sensitivity = P(warn | contains), false-negative rate = 1 - sensitivity,
specificity = P(no warn | not contains). The reference is the database itself, so these
numbers isolate the error caused by misclassification (database errors are a separate
question, addressed by the expert audit).

Warning policies:
  top1          allergens of the top-1 class (the deployed behaviour)
  top3_union    allergens of any of the top-3 classes (conservative alerting)
  abstain@95    top-1, but predictions whose temperature-scaled confidence is below the
                threshold that gives 95 % selective accuracy on the validation set are
                withheld ("not sure"); temperature and threshold are fitted on validation only;
                metrics on the accepted images + the share withheld
Images whose true class has no database entry are excluded (reported).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import metrics as M  # noqa: E402
from foodallergen import paths  # noqa: E402
from foodallergen.classes import SYNONYM_MERGES  # noqa: E402
from foodallergen.data import read_class_map  # noqa: E402
from foodallergen.experiments import SEEDS  # noqa: E402

REGULATED = list(range(1, 15))  # database ids 1..14 = the 14 regulated allergens


def class_profiles(db_dir, class_ids):
    foods = {r["YiyecekAdi"]: r["YiyecekID"] for r in csv.DictReader(open(db_dir / "foods.csv", encoding="utf-8"))}
    rel = defaultdict(set)
    for r in csv.DictReader(open(db_dir / "food_allergens.csv", encoding="utf-8")):
        rel[r["YiyecekID"]].add(int(r["AlerjenID"]))
    names = {r["AlerjenID"]: r["AlerjenAdi"] for r in csv.DictReader(open(db_dir / "allergens.csv", encoding="utf-8"))}
    reverse = defaultdict(list)
    for a, b in SYNONYM_MERGES.items():
        reverse[b].append(a)
    prof, covered = [], []
    for c in class_ids:
        entries = [n for n in [c, *reverse.get(c, [])] if n in foods]
        prof.append(set().union(*(rel[foods[n]] for n in entries)) if entries else set())
        covered.append(bool(entries))
    return prof, np.array(covered), {int(k): v for k, v in names.items()}


def evaluate(logits, labels, prof, covered, threshold=None, temperature=1.0):
    keep = covered[labels]
    lg, y = logits[keep], labels[keep]
    probs = M.calibrated_probs(lg, temperature)
    top = np.argsort(-lg, axis=1)
    res = {}
    for policy in ("top1", "top3_union", "abstain"):
        if policy == "abstain" and threshold is None:
            continue
        accepted = np.ones(len(y), bool)
        if policy == "abstain":
            accepted = probs.max(1) >= threshold
        out = {"images": int(keep.sum()), "withheld_share": float(1 - accepted.mean())}
        for a in REGULATED:
            truth = np.array([a in prof[t] for t in y])
            k = 3 if policy == "top3_union" else 1
            warn = np.array([any(a in prof[c] for c in row[:k]) for row in top])
            t, w = truth[accepted], warn[accepted]
            pos, neg = t.sum(), (~t).sum()
            out[a] = dict(positives=int(pos),
                          sensitivity=float((w & t).sum() / pos) if pos else None,
                          specificity=float((~w & ~t).sum() / neg) if neg else None,
                          false_negatives=int((~w & t).sum()))
        res[policy] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(paths.REPO_DATA / "allergen_db_v1"))
    ap.add_argument("--model", default="effv2s")
    ap.add_argument("--group", default="main", choices=["main", "final"])
    args = ap.parse_args()
    db = Path(args.db)
    classes = read_class_map()
    prof, covered, names = class_profiles(db, [c["class_id"] for c in classes])
    per_seed = []
    for s in SEEDS:
        rd = paths.RUNS / args.group / args.model / f"seed{s}"
        if not (rd / "DONE").exists():
            continue
        z = np.load(rd / "logits_test.npz")
        zv = np.load(rd / "logits_val.npz")
        vl, vy = zv["logits"].astype(np.float32), zv["labels"]
        T = M.fit_temperature(vl, vy)  # same rule as the deployed application (validation only)
        cov, sel, thr, _ = M.risk_coverage(M.calibrated_probs(vl, T), vy)
        th = M.coverage_at_accuracy(cov, sel, thr, 0.95)["threshold"]
        r = evaluate(z["logits"].astype(np.float32), z["labels"], prof, covered, th, T)
        r["threshold_95"], r["temperature"] = th, T
        per_seed.append(r)
    if not per_seed:
        sys.exit("no finished runs")
    rows = []
    for policy in ("top1", "top3_union", "abstain"):
        if policy not in per_seed[0]:
            continue
        for a in REGULATED:
            sens = [r[policy][a]["sensitivity"] for r in per_seed if r[policy][a]["sensitivity"] is not None]
            spec = [r[policy][a]["specificity"] for r in per_seed if r[policy][a]["specificity"] is not None]
            pos = per_seed[0][policy][a]["positives"]
            if not pos:
                continue
            ms, ss = M.mean_sd(sens)
            mp, sp = M.mean_sd(spec)
            rows.append({"Policy": policy, "Allergen": names[a], "Test images with allergen": pos,
                         "Sensitivity (%)": f"{ms * 100:.2f} ± {ss * 100:.2f}",
                         "False-negative rate (%)": f"{(1 - ms) * 100:.2f}",
                         "Specificity (%)": f"{mp * 100:.2f} ± {sp * 100:.2f}",
                         "Withheld (%)": f"{np.mean([r[policy]['withheld_share'] for r in per_seed]) * 100:.1f}"})
    out = paths.RUNS / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.group}_{Path(args.db).name}"
    with open(out / f"allergen_metrics_{args.model}_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / f"allergen_metrics_{args.model}_{tag}.json").write_text(
        json.dumps({"excluded_test_images_no_db_entry": int((~covered[np.load(
            paths.RUNS / args.group / args.model / f'seed{SEEDS[0]}' / 'logits_test.npz')['labels']]).sum()),
            "classes_without_db_entry": [c["class_id"] for c, ok in zip(classes, covered) if not ok],
            "per_seed": per_seed}, default=str, indent=1), encoding="utf-8")
    for r in rows:
        if r["Allergen"].startswith("Gluten"):
            print(r)


if __name__ == "__main__":
    main()
