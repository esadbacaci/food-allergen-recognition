# -*- coding: utf-8 -*-
"""Allergen consequences of classification errors and risk-aware allergen alerting.
No retraining: uses the saved validation/test logits and the allergen database.

    python scripts/evaluate_allergen_risk.py [--db data/allergen_db_v2] [--target 0.99]

Notation: p(c|x) = temperature-scaled class probabilities (temperature fitted on validation);
m[c, a] = 1 if the database profile of class c lists regulated allergen a (any presence level).

1. Error analysis. For every misclassified test image, the regulated-allergen profile of the
   predicted dish is compared with that of the true dish: identical (allergen-neutral error),
   at least one allergen missed (false negative), or only additional allergens (false alarm).
2. Risk-aware alerting. Allergen risk r_a(x) = sum_c p(c|x) m[c, a], i.e. the probability that
   the dish contains a according to the classifier and the database. Alert if r_a >= theta_a,
   where theta_a is the largest threshold with sensitivity >= target on the VALIDATION set.
   Tiered message: r_a >= 0.5 -> 'contains / may contain'; theta_a <= r_a < 0.5 -> 'uncertain:
   check the label'. Compared with the top-1 dish, the union of the top-3 dishes and top-1 with
   abstention (threshold for 95% selective accuracy on validation).
Outputs: 03_deneyler/analysis/allergen_risk_<group>_<db>.json, allergen_risk_final_<db>.csv,
         allergen_risk_backbones_<db>.csv, allergen_risk_thresholds_<db>.json (for the app)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import metrics as M  # noqa: E402
from foodallergen import paths  # noqa: E402
from foodallergen.data import read_class_map  # noqa: E402
from foodallergen.experiments import MODELS, SEEDS  # noqa: E402
from foodallergen.models import DISPLAY  # noqa: E402

REGULATED = list(range(1, 15))


def profile_matrix(db, class_ids):
    rows = lambda f: list(csv.DictReader(open(db / f, encoding="utf-8")))  # noqa: E731
    foods = {r["YiyecekID"]: r["YiyecekAdi"] for r in rows("foods.csv")}
    names = {int(r["AlerjenID"]): (r.get("AlerjenAdiEN") or r["AlerjenAdi"]) for r in rows("allergens.csv")}
    idx = {c: i for i, c in enumerate(class_ids)}
    m = np.zeros((len(class_ids), len(REGULATED)), bool)
    covered = np.zeros(len(class_ids), bool)
    for r in rows("food_allergens.csv"):
        a = int(r["AlerjenID"])
        c = foods[r["YiyecekID"]]
        if c in idx:
            covered[idx[c]] = True
            if a in REGULATED:
                m[idx[c], REGULATED.index(a)] = True
    for c in foods.values():
        if c in idx:
            covered[idx[c]] = True
    return m, covered, [names[a] for a in REGULATED]


def threshold_for_sensitivity(risk, truth, target):
    """Conformal threshold: the floor(alpha (n + 1))-th smallest risk among the n validation positives
    (alpha = 1 - target). For a test positive exchangeable with them, P(risk >= theta) >= target."""
    pos = np.sort(risk[truth])
    if len(pos) == 0:
        return None
    k = int(np.floor((1 - target) * (len(pos) + 1)))  # 1-indexed order statistic
    return float(pos[k - 1]) if k >= 1 else 0.0


def rates(warn, truth):
    pos, neg = truth.sum(), (~truth).sum()
    return dict(sens=float((warn & truth).sum() / pos) if pos else None,
                spec=float((~warn & ~truth).sum() / neg) if neg else None,
                fn=int((~warn & truth).sum()), pos=int(pos))


def evaluate_run(rd, m, covered, target):
    zv, zt = np.load(rd / "logits_val.npz"), np.load(rd / "logits_test.npz")
    vl, vy = zv["logits"].astype(np.float32), zv["labels"]
    tl, ty = zt["logits"].astype(np.float32), zt["labels"]
    kv, kt = covered[vy], covered[ty]
    vl, vy, tl, ty = vl[kv], vy[kv], tl[kt], ty[kt]
    T = M.fit_temperature(vl, vy)
    pv, pt = M.calibrated_probs(vl, T), M.calibrated_probs(tl, T)
    mf = m.astype(np.float64)
    rv, rt = pv @ mf, pt @ mf                     # allergen risk, images x allergens
    truth_v, truth_t = m[vy], m[ty]
    pred = pt.argmax(1)
    top3 = np.argsort(-pt, 1)[:, :3]
    cov, sel, thr, _ = M.risk_coverage(pv, vy)
    tau = M.coverage_at_accuracy(cov, sel, thr, 0.95)["threshold"]
    accepted = pt.max(1) >= tau

    # ---- 1. error analysis
    wrong = pred != ty
    t_set, p_set = m[ty[wrong]], m[pred[wrong]]
    missed = (t_set & ~p_set).any(1)
    extra = (p_set & ~t_set).any(1)
    err = dict(n_test=int(len(ty)), n_errors=int(wrong.sum()), error_rate=float(wrong.mean()),
               neutral=float((~missed & ~extra).mean()), missed_any=float(missed.mean()),
               only_extra=float((~missed & extra).mean()),
               images_with_missed_allergen=float(missed.sum() / len(ty)),
               per_allergen_missed_share_of_errors={})
    for j in range(len(REGULATED)):
        err["per_allergen_missed_share_of_errors"][j] = float((t_set[:, j] & ~p_set[:, j]).mean())

    # ---- 2. policies per allergen
    out = {"temperature": T, "tau": tau, "withheld": float(1 - accepted.mean()), "errors": err, "allergens": {}}
    theta = {}
    any_miss = {k: np.zeros(len(ty), bool) for k in ("top1", "top3_union", "abstain", "risk")}
    check_label = np.zeros(len(ty), bool)
    for j in range(len(REGULATED)):
        tr = truth_t[:, j]
        if tr.sum() == 0 or truth_v[:, j].sum() == 0:
            continue
        th = threshold_for_sensitivity(rv[:, j], truth_v[:, j], target)
        theta[j] = th
        warn = {"top1": m[pred, j], "top3_union": m[top3, j].any(1), "risk": rt[:, j] >= th}
        res = {k: rates(w, tr) for k, w in warn.items()}
        res["abstain"] = rates(m[pred, j][accepted], tr[accepted])
        res["risk"]["check_label_share"] = float(((rt[:, j] >= th) & (rt[:, j] < 0.5)).mean())
        res["risk_auroc"] = float(roc_auc_score(tr, rt[:, j]))
        res["theta"] = th
        out["allergens"][j] = res
        for k, w in warn.items():
            any_miss[k] |= tr & ~w
        any_miss["abstain"] |= tr & ~m[pred, j] & accepted
        check_label |= (rt[:, j] >= th) & (rt[:, j] < 0.5)
    out["images_any_allergen_missed"] = {k: float(v.mean()) for k, v in any_miss.items()}
    out["images_any_allergen_missed"]["abstain"] = float(any_miss["abstain"][accepted].mean())
    out["images_with_check_label"] = float(check_label.mean())
    out["theta"] = theta
    return out


def summarize(per_seed, names):
    rows = []
    for j in per_seed[0]["allergens"]:
        a = [s["allergens"][j] for s in per_seed]
        mean = lambda f: float(np.mean([f(x) for x in a]))  # noqa: E731
        rows.append({
            "Allergen": names[j], "Test images": a[0]["top1"]["pos"],
            "Top-1 FNR (%)": round(100 * (1 - mean(lambda x: x["top1"]["sens"])), 2),
            "Top-1 Spec (%)": round(100 * mean(lambda x: x["top1"]["spec"]), 2),
            "Top-3 union FNR (%)": round(100 * (1 - mean(lambda x: x["top3_union"]["sens"])), 2),
            "Top-3 union Spec (%)": round(100 * mean(lambda x: x["top3_union"]["spec"]), 2),
            "Abstention FNR (%)": round(100 * (1 - mean(lambda x: x["abstain"]["sens"])), 2),
            "Abstention Spec (%)": round(100 * mean(lambda x: x["abstain"]["spec"]), 2),
            "Risk FNR (%)": round(100 * (1 - mean(lambda x: x["risk"]["sens"])), 2),
            "Risk FNR SD": round(100 * float(np.std([x["risk"]["sens"] for x in a], ddof=1)), 2),
            "Risk Spec (%)": round(100 * mean(lambda x: x["risk"]["spec"]), 2),
            "Check-label share (%)": round(100 * mean(lambda x: x["risk"]["check_label_share"]), 2),
            "Risk AUROC": round(mean(lambda x: x["risk_auroc"]), 4),
            "theta (mean)": round(mean(lambda x: x["theta"]), 4),
        })
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(paths.REPO_DATA / "allergen_db_v2"))
    ap.add_argument("--target", type=float, default=0.99)
    args = ap.parse_args()
    db = Path(args.db)
    class_ids = [c["class_id"] for c in read_class_map()]
    m, covered, names = profile_matrix(db, class_ids)
    out_dir = paths.RUNS / "analysis"
    tag = db.name
    result = {"target_sensitivity": args.target, "allergens": names, "groups": {}}

    # deployed model (final group)
    final = [evaluate_run(paths.RUNS / "final" / "maxvit_t" / f"seed{s}", m, covered, args.target) for s in SEEDS]
    result["groups"]["final/maxvit_t"] = final
    write_csv(out_dir / f"allergen_risk_final_{tag}.csv", summarize(final, names))

    # all backbones of the controlled comparison
    rows = []
    gl = names.index(next(n for n in names if n.startswith(("Cereals", "Gluten"))))
    for mdl in MODELS:
        runs = [evaluate_run(paths.RUNS / "main" / mdl / f"seed{s}", m, covered, args.target) for s in SEEDS]
        result["groups"][f"main/{mdl}"] = runs
        f = lambda g: float(np.mean([g(r) for r in runs]))  # noqa: E731
        rows.append({
            "Model": DISPLAY[mdl],
            "Top-1 error (%)": round(100 * f(lambda r: r["errors"]["error_rate"]), 2),
            "Allergen-neutral errors (%)": round(100 * f(lambda r: r["errors"]["neutral"]), 1),
            "Errors missing >=1 allergen (%)": round(100 * f(lambda r: r["errors"]["missed_any"]), 1),
            "Images with a missed allergen, top-1 (%)": round(100 * f(lambda r: r["images_any_allergen_missed"]["top1"]), 2),
            "Images with a missed allergen, risk (%)": round(100 * f(lambda r: r["images_any_allergen_missed"]["risk"]), 2),
            "Gluten FNR top-1 (%)": round(100 * (1 - f(lambda r: r["allergens"][gl]["top1"]["sens"])), 2),
            "Gluten FNR risk (%)": round(100 * (1 - f(lambda r: r["allergens"][gl]["risk"]["sens"])), 2),
            "Gluten spec risk (%)": round(100 * f(lambda r: r["allergens"][gl]["risk"]["spec"]), 2),
            "Images with check-label (%)": round(100 * f(lambda r: r["images_with_check_label"]), 1),
        })
    write_csv(out_dir / f"allergen_risk_backbones_{tag}.csv", rows)
    (out_dir / f"allergen_risk_{tag}.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

    # thresholds of the deployed seed (highest validation accuracy) for the application
    vals = {s: json.loads((paths.RUNS / "final" / "maxvit_t" / f"seed{s}" / "metrics.json").read_text())["val"]["acc"]
            for s in SEEDS}
    best = max(vals, key=vals.get)
    dep = final[SEEDS.index(best)]
    codes = {int(r["AlerjenID"]): r.get("Kod", "") for r in csv.DictReader(open(db / "allergens.csv", encoding="utf-8"))}
    (out_dir / f"allergen_risk_thresholds_{tag}.json").write_text(json.dumps(
        dict(run=f"final/maxvit_t/seed{best}", target_sensitivity=args.target, likely_threshold=0.5,
             thresholds={codes.get(REGULATED[int(j)], str(REGULATED[int(j)])): v for j, v in dep["theta"].items()}),
        indent=2), encoding="utf-8")
    for r in rows:
        print(r)
    print(open(out_dir / f"allergen_risk_final_{tag}.csv", encoding="utf-8").read())
    print({k: round(v * 100, 2) for k, v in final[0]["images_any_allergen_missed"].items()},
          "check-label images %", round(final[0]["images_with_check_label"] * 100, 1))


if __name__ == "__main__":
    main()
