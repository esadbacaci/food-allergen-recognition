# -*- coding: utf-8 -*-
"""Out-of-distribution (non-food) evaluation: the deployed food gate and the classifier's
confidence, on in-distribution test images vs. Caltech-101 non-food images (reviewers 2, 3).

    python scripts/evaluate_ood.py --part gate
    python scripts/evaluate_ood.py --part classifier --group main      (after training)
    python scripts/evaluate_ood.py --part classifier --group final

OOD set: Caltech-101 [Fei-Fei et al., CVPR-W 2004; CaltechDATA, CC BY 4.0] without the
categories that show food or drink (EXCLUDED) and without the unlabeled BACKGROUND_Google
clutter class. The file list is written to data/ood_caltech101.csv.

Food gate = the mechanism of the web application: ImageNet-1K EfficientNet-B0 (torchvision
weights), sum of the softmax probabilities of 72 food/serving-related ImageNet classes,
image stretched to 224x224 as in ImageTensorizer.cs, threshold 0.12 (appsettings).
Outputs: 03_deneyler/analysis/ood_gate.json, ood_classifier_<group>.json
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tvm
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import metrics as M  # noqa: E402
from foodallergen import paths  # noqa: E402
from foodallergen.data import MEAN, STD, read_class_map, read_manifest  # noqa: E402
from foodallergen.models import build_model  # noqa: E402

OOD_ROOT = paths.SOURCE / "ood" / "101_ObjectCategories"
EXCLUDED = {"pizza", "strawberry", "lobster", "crab", "crayfish", "cup", "bass", "octopus",
            "BACKGROUND_Google"}
GATE_CLASSES = paths.REPO_DATA / "food_gate_classes.txt"  # copy of calorieapp/ml/food_gate_classes.txt
GATE_THRESHOLD = 0.12
OUT = paths.RUNS / "analysis"


class Files(Dataset):
    def __init__(self, files, tf):
        self.files, self.tf = files, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            return self.tf(im.convert("RGB"))


def ood_files():
    files, rows = [], []
    for d in sorted(OOD_ROOT.iterdir()):
        if d.is_dir() and d.name not in EXCLUDED:
            for f in sorted(d.glob("*.jpg")):
                files.append(f)
                rows.append(dict(category=d.name, file=f.name))
    paths.REPO_DATA.mkdir(exist_ok=True)
    with open(paths.REPO_DATA / "ood_caltech101.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["category", "file"])
        w.writeheader()
        w.writerows(rows)
    return files


def id_test_files(image_dir="images"):
    return [paths.V2_DIR / image_dir / r["path"] for r in read_manifest() if r["split"] == "test"]


@torch.no_grad()
def run_model(model, files, tf, bs=128, workers=2):
    model = model.cuda().eval()
    outs = []
    for x in DataLoader(Files(files, tf), batch_size=bs, num_workers=workers):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(model(x.cuda(non_blocking=True)).float().cpu())
    return torch.cat(outs).numpy()


def fpr_at_95_tpr(id_scores, ood_scores):
    thr = np.percentile(id_scores, 5)  # 95 % of ID images accepted
    return float((ood_scores >= thr).mean())


def part_gate():
    idx = [int(re.split(r"\s|#", line.strip())[0]) for line in GATE_CLASSES.read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.startswith("#")]
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    res = {}
    scores = {}
    for name, files in [("id_test", id_test_files()), ("ood_caltech101", ood_files())]:
        p = torch.softmax(torch.from_numpy(run_model(model, files, tf)), 1).numpy()
        scores[name] = p[:, idx].sum(1)
    s_id, s_ood = scores["id_test"], scores["ood_caltech101"]
    y = np.r_[np.ones(len(s_id)), np.zeros(len(s_ood))]
    res = dict(gate_classes=len(idx), threshold=GATE_THRESHOLD, n_id=len(s_id), n_ood=len(s_ood),
               auroc=float(roc_auc_score(y, np.r_[s_id, s_ood])),
               id_rejected_at_threshold=float((s_id < GATE_THRESHOLD).mean()),
               ood_accepted_at_threshold=float((s_ood >= GATE_THRESHOLD).mean()),
               fpr_at_95_tpr=fpr_at_95_tpr(s_id, s_ood),
               excluded_caltech_categories=sorted(EXCLUDED))
    # per-source false rejection (Food-101 vs Turkish test images)
    src = np.array([r["source"] for r in read_manifest() if r["split"] == "test"])
    res["id_rejected_food101"] = float((s_id[src == "food101"] < GATE_THRESHOLD).mean())
    res["id_rejected_turkish"] = float((s_id[src != "food101"] < GATE_THRESHOLD).mean())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ood_gate.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(OUT / "ood_gate_scores.npz", id=s_id, ood=s_ood)
    print(json.dumps(res, indent=2))


def part_classifier(group):
    n_cls = len(read_class_map())
    files = ood_files()
    res = {}
    run_dirs = sorted((paths.RUNS / group).glob("*/seed*"))
    for rd in run_dirs:
        if not (rd / "best.pt").exists():
            continue
        cfg = json.loads((rd / "config.json").read_text())
        size = cfg.get("img_size", 224)
        tf = T.Compose([T.Resize(int(round(size / 0.875)), interpolation=T.InterpolationMode.BICUBIC),
                        T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
        model = build_model(cfg["model"], n_cls, pretrained=False)
        model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu"))
        ood_logits = run_model(model, files, tf)
        z = np.load(rd / "logits_test.npz")
        zv = np.load(rd / "logits_val.npz")
        id_logits = z["logits"].astype(np.float32)
        T_ = M.fit_temperature(zv["logits"].astype(np.float32), zv["labels"])
        msp_id = M.calibrated_probs(id_logits, T_).max(1)
        msp_ood = M.calibrated_probs(ood_logits, T_).max(1)
        en_id = torch.logsumexp(torch.from_numpy(id_logits), 1).numpy()
        en_ood = torch.logsumexp(torch.from_numpy(ood_logits), 1).numpy()
        cov, sel, thr, _ = M.risk_coverage(M.calibrated_probs(zv["logits"].astype(np.float32), T_), zv["labels"])
        tau = M.coverage_at_accuracy(cov, sel, thr, 0.95)["threshold"]
        y = np.r_[np.ones(len(msp_id)), np.zeros(len(msp_ood))]
        res[f"{rd.parent.name}/{rd.name}"] = dict(
            temperature=T_, auroc_msp=float(roc_auc_score(y, np.r_[msp_id, msp_ood])),
            auroc_energy=float(roc_auc_score(y, np.r_[en_id, en_ood])),
            fpr95_msp=fpr_at_95_tpr(msp_id, msp_ood), fpr95_energy=fpr_at_95_tpr(en_id, en_ood),
            tau_95_selective=tau,
            ood_accepted_at_tau=float((msp_ood >= tau).mean()) if tau is not None else None,
            id_accepted_at_tau=float((msp_id >= tau).mean()) if tau is not None else None)
        print(rd, res[f"{rd.parent.name}/{rd.name}"], flush=True)
    (OUT / f"ood_classifier_{group}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["gate", "classifier"])
    ap.add_argument("--group", default="main", choices=["main", "final"])
    a = ap.parse_args()
    part_gate() if a.part == "gate" else part_classifier(a.group)
