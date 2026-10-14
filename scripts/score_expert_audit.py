# -*- coding: utf-8 -*-
"""Score the database against the blind expert labels.

    python scripts/score_expert_audit.py <filled form .xlsx> [--db v1|v2]

A relation counts as present when marked "Her zaman" (always) or "Sıklıkla" (often);
"Olabilir" (possible) is analysed separately as a may-contain warning.
Reports, per regulated allergen and pooled: precision, recall (sensitivity), false-negative
count, Cohen's kappa, and observed agreement. Output: data/allergen_db_v2/expert_audit/score_<db>.json
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import load_workbook
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.classes import SYNONYM_MERGES  # noqa: E402

CODES = ["GLU", "CRU", "EGG", "FSH", "PNT", "SOY", "MLK", "NUT", "CEL", "MUS", "SES", "SUL", "LUP", "MOL"]
PRESENT = {"Her zaman", "Sıklıkla"}
V1_CODE = {1: "GLU", 2: "MLK", 3: "EGG", 4: "PNT", 5: "NUT", 6: "SOY", 7: "FSH", 8: "CRU", 9: "SES",
           10: "CEL", 11: "MUS", 12: "SUL", 13: "LUP", 14: "MOL"}
D2 = paths.REPO_DATA / "allergen_db_v2"


def db_profiles(which):
    if which == "v1":
        d1 = paths.REPO_DATA / "allergen_db_v1"
        foods = {r["YiyecekID"]: r["YiyecekAdi"] for r in csv.DictReader(open(d1 / "foods.csv", encoding="utf-8"))}
        prof = defaultdict(set)
        for r in csv.DictReader(open(d1 / "food_allergens.csv", encoding="utf-8")):
            code = V1_CODE.get(int(r["AlerjenID"]))
            if code:
                prof[foods[r["YiyecekID"]]].add(code)
        for a, b in SYNONYM_MERGES.items():
            prof[b] |= prof.get(a, set())
        return prof
    draft = json.load(open(D2 / "llm_draft_v2.json", encoding="utf-8"))
    return {e["class_id"]: {k for k, v in e["allergens"].items() if v in ("always", "often")}
            for e in draft}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("form")
    ap.add_argument("--db", default="v2", choices=["v1", "v2"])
    args = ap.parse_args()
    sample = list(csv.DictReader(open(D2 / "expert_audit" / "sample.csv", encoding="utf-8")))
    ws = load_workbook(args.form, data_only=True)["Form"]
    expert = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        cid = sample[int(row[0]) - 1]["class_id"]
        expert[cid] = {c: (row[4 + k] or "").strip() for k, c in enumerate(CODES)}
    prof = db_profiles(args.db)
    res, all_e, all_d = {}, [], []
    for c in CODES:
        e = [int(expert[d][c] in PRESENT) for d in expert]
        p = [int(c in prof.get(d, set())) for d in expert]
        tp = sum(a and b for a, b in zip(e, p))
        fp = sum(b and not a for a, b in zip(e, p))
        fn = sum(a and not b for a, b in zip(e, p))
        res[c] = dict(expert_positive=sum(e), db_positive=sum(p), tp=tp, fp=fp, fn=fn,
                      precision=tp / (tp + fp) if tp + fp else None,
                      recall=tp / (tp + fn) if tp + fn else None,
                      kappa=float(cohen_kappa_score(e, p)) if len(set(e + p)) > 1 else None,
                      agreement=sum(a == b for a, b in zip(e, p)) / len(e))
        all_e += e
        all_d += p
    tp = sum(a and b for a, b in zip(all_e, all_d))
    fp = sum(b and not a for a, b in zip(all_e, all_d))
    fn = sum(a and not b for a, b in zip(all_e, all_d))
    res["pooled"] = dict(decisions=len(all_e), tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp),
                         recall=tp / (tp + fn), kappa=float(cohen_kappa_score(all_e, all_d)),
                         agreement=sum(a == b for a, b in zip(all_e, all_d)) / len(all_e))
    out = D2 / "expert_audit" / f"score_{args.db}.json"
    out.write_text(json.dumps(dict(dishes=len(expert), per_allergen=res), indent=2), encoding="utf-8")
    print(json.dumps(res["pooled"], indent=2), "\nGLU:", res["GLU"])


if __name__ == "__main__":
    main()
