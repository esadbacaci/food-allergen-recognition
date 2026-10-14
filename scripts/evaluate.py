"""Tables and figures for the revised article, computed from saved logits.

    python scripts/evaluate.py            -> 03_deneyler/analysis/

Outputs (English labels, as required for the article):
  main_per_run.csv         every run: overall / Food-101 official test (101-way) / Turkish subset
  main_table.csv|.md       mean +- sample SD over seeds per backbone
  mcnemar_holm.csv         all backbone pairs, every seed, Holm-adjusted p-values
  ablation_table.csv|.md   recipe ablation (mean +- SD, difference to the full recipe)
  calibration.json         temperature scaling, ECE, NLL, risk-coverage summary per backbone
  risk_coverage.png        selective accuracy vs coverage (seed 1, calibrated)
  per_class_f1_<model>.csv and top_confusions_<model>.csv for the best backbone
"""
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import metrics as M  # noqa: E402
from foodallergen import paths  # noqa: E402
from foodallergen.data import read_class_map, read_manifest  # noqa: E402
from foodallergen.experiments import ABLATIONS, ABLATION_MODEL, MODELS, SEEDS  # noqa: E402
from foodallergen.models import DISPLAY  # noqa: E402

OUT = paths.RUNS / "analysis"


def load(run_dir, split="test"):
    z = np.load(run_dir / f"logits_{split}.npz")
    return z["logits"].astype(np.float32), z["labels"]


def fmt(mean, sd, pct=True):
    k = 100 if pct else 1
    return f"{mean * k:.2f} ± {sd * k:.2f}" if pct else f"{mean:.4f} ± {sd:.4f}"


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_md(path, header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    classes = read_class_map()
    n_cls = len(classes)
    f101_idx = [int(c["index"]) for c in classes if c["group"] == "food101"]
    raw_idx = {int(c["index"]) for c in classes if c["raw_produce"] == "1"}
    test_rows = [r for r in read_manifest() if r["split"] == "test"]
    src = np.array([r["source"] for r in test_rows])
    is_f101 = src == "food101"
    is_tr = ~is_f101

    # ------------------------------------------------------------- main comparison
    per_run, by_model, correct = [], defaultdict(list), {}
    for m in MODELS:
        for s in SEEDS:
            rd = paths.RUNS / "main" / m / f"seed{s}"
            if not (rd / "DONE").exists():
                continue
            lg, y = load(rd)
            info = json.loads((rd / "metrics.json").read_text())
            tr_dish = is_tr & ~np.isin(y, list(raw_idx))
            row = dict(model=m, seed=s, best_epoch=info["best_epoch"],
                       mean_epoch_seconds=info["mean_epoch_seconds"],
                       peak_gpu_mem_mb=info["peak_gpu_mem_mb"])
            for name, kw in [("all", {}), ("f101_official_101way", dict(mask=is_f101, restrict_to=f101_idx)),
                             ("turkish", dict(mask=is_tr)), ("turkish_dishes_only", dict(mask=tr_dish))]:
                for k, v in M.classification(lg, y, **kw).items():
                    row[f"{name}_{k}"] = v
            per_run.append(row)
            by_model[m].append(row)
            correct[(m, s)] = lg.argmax(1) == y
    if not per_run:
        sys.exit("no finished main runs")
    write_csv(OUT / "main_per_run.csv", per_run)

    table = []
    for m in MODELS:
        rs = by_model.get(m)
        if not rs:
            continue
        t = {"Model": DISPLAY[m], "Seeds": len(rs)}
        for col, key in [("Top-1 (%)", "all_acc"), ("Top-5 (%)", "all_top5"),
                         ("Macro-F1 (%)", "all_macro_f1"), ("Macro-P (%)", "all_macro_precision"),
                         ("Macro-R (%)", "all_macro_recall"),
                         ("Food-101 official test, 101-way (%)", "f101_official_101way_acc"),
                         ("Turkish subset (%)", "turkish_acc"),
                         ("Turkish dishes w/o raw produce (%)", "turkish_dishes_only_acc")]:
            t[col] = fmt(*M.mean_sd([r[key] for r in rs]))
        t["Best epoch"] = f"{np.mean([r['best_epoch'] for r in rs]):.1f}"
        table.append(t)
    write_csv(OUT / "main_table.csv", table)
    write_md(OUT / "main_table.md", list(table[0]), table)

    # ------------------------------------------------------------- final (deployed) model
    final_rows = []
    for rd in sorted((paths.RUNS / "final").glob("*/seed*")):
        if (rd / "DONE").exists():
            lg, y = load(rd)
            row = dict(model=rd.parent.name, seed=rd.name)
            for name, kw in [("all", {}), ("f101_official_101way", dict(mask=is_f101, restrict_to=f101_idx)),
                             ("turkish", dict(mask=is_tr))]:
                for k, v in M.classification(lg, y, **kw).items():
                    row[f"{name}_{k}"] = v
            final_rows.append(row)
    if final_rows:
        write_csv(OUT / "final_per_run.csv", final_rows)
        ft = [{"Model": DISPLAY[final_rows[0]["model"]] + " (320 px, 30 epochs)", "Seeds": len(final_rows),
               **{col: fmt(*M.mean_sd([r[key] for r in final_rows])) for col, key in
                  [("Top-1 (%)", "all_acc"), ("Top-5 (%)", "all_top5"), ("Macro-F1 (%)", "all_macro_f1"),
                   ("Food-101 official test, 101-way (%)", "f101_official_101way_acc"),
                   ("Turkish subset (%)", "turkish_acc")]}}]
        write_csv(OUT / "final_table.csv", ft)
        write_md(OUT / "final_table.md", list(ft[0]), ft)

    # ------------------------------------------------------------- McNemar + Holm
    mc_rows = []
    for s in SEEDS:
        pairs = [(a, b) for a, b in itertools.combinations(MODELS, 2)
                 if (a, s) in correct and (b, s) in correct]
        if not pairs:
            continue
        res = [M.mcnemar(correct[(a, s)], correct[(b, s)]) for a, b in pairs]
        adj = M.holm([r["p"] for r in res])
        for (a, b), r, pa in zip(pairs, res, adj):
            mc_rows.append(dict(seed=s, model_a=DISPLAY[a], model_b=DISPLAY[b], **r,
                                p_holm=float(pa), significant_005=bool(pa < 0.05)))
    if mc_rows:
        write_csv(OUT / "mcnemar_holm.csv", mc_rows)

    # ------------------------------------------------------------- ablation
    ab = []
    full = [r["all_acc"] for r in by_model.get(ABLATION_MODEL, [])]
    full_f1 = [r["all_macro_f1"] for r in by_model.get(ABLATION_MODEL, [])]
    variants = [("full_recipe", None)] + list(ABLATIONS.items())
    for name, _ in variants:
        if name == "full_recipe":
            accs, f1s = full, full_f1
        else:
            accs, f1s = [], []
            for s in SEEDS:
                rd = paths.RUNS / "ablation" / name / f"seed{s}"
                if (rd / "DONE").exists():
                    lg, y = load(rd)
                    c = M.classification(lg, y)
                    accs.append(c["acc"])
                    f1s.append(c["macro_f1"])
        if accs:
            d = (np.mean(accs) - np.mean(full)) * 100 if full else float("nan")
            ab.append({"Variant": name, "Seeds": len(accs), "Top-1 (%)": fmt(*M.mean_sd(accs)),
                       "Macro-F1 (%)": fmt(*M.mean_sd(f1s)), "Δ Top-1 vs full (pp)": f"{d:+.2f}"})
    if len(ab) > 1:
        write_csv(OUT / "ablation_table.csv", ab)
        write_md(OUT / "ablation_table.md", list(ab[0]), ab)

    # ------------------------------------------------------------- calibration
    calib = {}
    curves = {}
    for m in MODELS:
        out = []
        for s in SEEDS:
            rd = paths.RUNS / "main" / m / f"seed{s}"
            if not (rd / "DONE").exists():
                continue
            vl, vy = load(rd, "val")
            tl, ty = load(rd, "test")
            T = M.fit_temperature(vl, vy)
            p0, p1 = M.calibrated_probs(tl), M.calibrated_probs(tl, T)
            cov, sel, thr, aurc = M.risk_coverage(p1, ty)
            out.append(dict(seed=s, temperature=T, ece_before=M.ece(p0, ty), ece_after=M.ece(p1, ty),
                            nll_before=M.nll(p0, ty), nll_after=M.nll(p1, ty), aurc=aurc,
                            **{f"coverage_at_{int(t * 100)}": M.coverage_at_accuracy(cov, sel, thr, t)
                               for t in (0.90, 0.95, 0.98)}))
            if s == SEEDS[0]:
                curves[m] = (cov, sel)
        if out:
            calib[m] = out
    (OUT / "calibration.json").write_text(json.dumps(calib, indent=2))
    if curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
        for m, (cov, sel) in curves.items():
            ax.plot(cov * 100, sel * 100, label=DISPLAY[m], lw=1.4)
        ax.set_xlabel("Coverage (%)")
        ax.set_ylabel("Selective top-1 accuracy (%)")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "risk_coverage.png")

    # ------------------------------------------------------------- best backbone details
    best = max(by_model, key=lambda m: np.mean([r["all_acc"] for r in by_model[m]]))
    f1s, conf = [], np.zeros((n_cls, n_cls), int)
    for s in SEEDS:
        rd = paths.RUNS / "main" / best / f"seed{s}"
        if (rd / "DONE").exists():
            lg, y = load(rd)
            f1s.append(M.per_class_f1(lg, y, n_cls))
            np.add.at(conf, (y, lg.argmax(1)), 1)
    f1m = np.mean(f1s, axis=0)
    write_csv(OUT / f"per_class_f1_{best}.csv",
              [dict(class_id=c["class_id"], display_en=c["display_en"], group=c["group"],
                    raw_produce=c["raw_produce"], f1=round(float(f1m[i]), 4))
               for i, c in enumerate(classes)])
    np.fill_diagonal(conf, 0)
    top = np.dstack(np.unravel_index(np.argsort(-conf, axis=None)[:25], conf.shape))[0]
    write_csv(OUT / f"top_confusions_{best}.csv",
              [dict(true=classes[i]["display_en"], predicted=classes[j]["display_en"],
                    errors_summed_over_seeds=int(conf[i, j]),
                    true_group=classes[i]["group"], predicted_group=classes[j]["group"])
               for i, j in top])
    print(open(OUT / "main_table.md", encoding="utf-8").read())
    print("best backbone:", DISPLAY[best], "| outputs in", OUT)


if __name__ == "__main__":
    main()
