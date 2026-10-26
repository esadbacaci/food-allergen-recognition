# Food image classification linked to an allergen database (Food-101 + Turkish cuisine)

Code, data splits, allergen database and results for the article
*"Beyond Accuracy: Allergen-Level Evaluation and Risk-Aware Alerting for Food Image Classification Linked to an Allergen Database"* (M. E. Bacacı and B. Takgil, Düzce University; Applied Sciences, under revision).

The system has two parts with an explicit boundary:

1. **Image classifier** – a deep network assigns one of 206 dish classes to a photograph (the only learned component).
2. **Allergen lookup** – the typical allergen profile of the dish is retrieved from a versioned relational
   database. Allergens are **not** detected visually. A **risk-aware rule** sums the calibrated probabilities of all
   dishes whose profile contains an allergen and reports it, if necessary as "check the label", above a per-allergen
   conformal threshold (target sensitivity 99 % on the validation set).

The system is an informational aid. It does not replace reading the product label or the advice of a physician or
dietitian. Live demo: https://foodrecognition.com.tr

## Main results (test set, 31,747 images, mean ± SD over 3 seeds)

| | Value |
|---|---|
| Best backbone (controlled comparison of 7) | MaxViT-T, 91.91 ± 0.09 % top-1 |
| Deployed model (MaxViT-T, 30 epochs) | 92.12 ± 0.10 % top-1; 92.25 % on the official Food-101 test set (101-way) |
| Images with a missed regulated allergen, top-1 dish → risk-aware rule | 5.3 % → 1.4 % |
| Gluten false-negative rate, top-1 → risk-aware (specificity) | 1.86 % → 0.67 % (91.7 %) |
| Blind dietitian audit of the database (60 dishes, 840 decisions, v2.0) | precision 94.6 %, recall 93.5 %, κ = 0.93 |

## Repository layout

| Path | Content |
|---|---|
| `foodallergen/` | library: paths, label system, data loading, models, training loop, metrics |
| `scripts/build_dataset_v2.py` | dataset from the official Food-101 archive and the Turkish images (label repair, pHash deduplication, splits) |
| `scripts/run.py`, `scripts/run_all.sh` | all experiment groups (`sweep`, `main`, `final`, `ablation`) defined in `foodallergen/experiments.py` |
| `scripts/select_lr.py`, `scripts/select_final.py` | learning rate and deployed backbone, chosen on the validation set only |
| `scripts/evaluate.py` | main tables, McNemar + Holm, ablation, calibration, risk–coverage, per-class F1, confusions |
| `scripts/evaluate_allergen.py` | allergen-level sensitivity / specificity (top-1, top-3 union, abstention) |
| `scripts/evaluate_allergen_risk.py` | allergen consequences of errors and the risk-aware rule (conformal thresholds) |
| `scripts/evaluate_ood.py` | non-food (Caltech-101) test of the classifier and of the food gate |
| `scripts/benchmark_inference.py`, `scripts/export_onnx.py` | efficiency; ONNX export verified on the whole test set |
| `scripts/check_near_duplicates.py` | Turkish test accuracy without test images that have a same-class training look-alike at looser pHash thresholds |
| `scripts/allergen_strict_reference.py` | allergen metrics with the stricter reference of the dietitian audit (only 'always' / 'often' count as present) |
| `scripts/conformal_split_check.py` | risk-aware rule with temperature and thresholds fitted on disjoint halves of the validation set |
| `scripts/label_check_form.py` | label check of two images per Turkish class by a second person |
| `scripts/finalize_allergen_db_v2.py`, `scripts/apply_expert_audit.py`, `scripts/score_expert_audit.py` | database build, dietitian audit and its corrections |
| `data/manifest_v2.csv` | every image: class, source, split, MD5, perceptual hash, original file name |
| `data/class_map_v2.csv` | class index, English / Turkish names, group, per-split counts |
| `data/allergen_taxonomy_v2.csv` | 22 categories mapped to Reg. (EU) 1169/2011 Annex II and the Turkish Food Codex |
| `data/allergen_db_v2/` | database v2.1: CSVs, MySQL/MariaDB migrations, LLM prompt (`PROMPT.md`), raw LLM draft, rule-check log, adjudication, dietitian audit scores and corrections |
| `data/allergen_db_v1/` | database of the submitted (first) version |
| `v1_submission_logs/` | training logs of the experiments of the first submission |

## Data

Food-101 is available from its authors. The Turkish images are **not** redistributed: the Kaggle "Turkish Food"
collection has no declared licence and the images collected by the authors remain subject to the rights of their
owners. The manifest (file name, MD5, pHash, split) allows the Kaggle part to be rebuilt and the split to be verified.

## Weights

The trained deployed model (ONNX, opset 17, 224 × 224 input, 206 classes) is attached to the GitHub release
together with `deploy_config.json` (temperature, abstention threshold) and the risk thresholds. Because part of the
training data has no declared licence, the weights are released **for non-commercial research use only**.

## Paths

All paths derive from the project root (`foodallergen/paths.py`): by default the parent folder of this repository,
containing `01_veri/` (data) and `03_deneyler/` (runs). Set `FOOD_PROJECT_ROOT` to use another location.

## Licence

Code: MIT (see `LICENSE`). Weights: non-commercial research use only. Allergen database: CC BY 4.0.
