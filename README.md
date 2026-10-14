# Food recognition with allergen lookup (Food-101 + Turkish cuisine)

Code, data splits and allergen database for the article
*"Food Classification and Allergen Matching Using Deep Learning"* (Bacacı & Takgil, Düzce University).

The system has two parts with a clear boundary:

1. **Image classifier** – a deep network assigns one of the dish classes to a photograph.
   This is the only learned component.
2. **Allergen lookup** – the predicted class is looked up in a relational database that stores the
   *typical* allergen profile of the dish. Allergens are **not** detected visually.

The system is an informational aid. It does not replace reading the product label or the advice of a
physician or dietitian.

## Repository layout

| Path | Content |
|---|---|
| `foodallergen/` | library code: paths, label system, data loading, models, training loop, metrics |
| `scripts/build_dataset_v2.py` | builds the dataset from the official Food-101 archive and the Turkish images (label repair, deduplication, splits) |
| `scripts/run.py` | runs the experiment groups defined in `foodallergen/experiments.py` (`sweep`, `main`, `ablation`) |
| `scripts/select_lr.py` | picks each backbone's learning rate from the validation sweep |
| `scripts/evaluate.py` | all tables / figures from the saved logits |
| `scripts/benchmark_inference.py` | parameters, FLOPs, GPU and CPU (ONNX Runtime) latency under identical conditions |
| `scripts/export_allergen_db.py` | exports the MySQL allergen database to CSV |
| `data/manifest_v2.csv` | every image: class, source, split, MD5, pHash |
| `data/class_map_v2.csv` | class index, English / Turkish display names, group, per-split counts |
| `data/dataset_report_v2.json` | what the dataset build changed and why (counts) |
| `data/allergen_db_v1/` | allergen database export |

## Paths

All paths derive from the project root (`foodallergen/paths.py`): by default the parent folder of this
repository, which contains `01_veri/` (data), `03_deneyler/` (runs) and `04_alerjen_db/`.
Set `FOOD_PROJECT_ROOT` to use another location. No absolute path appears in the code.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/build_dataset_v2.py --stage index_v1     # Turkish images (v1 folder)
python scripts/build_dataset_v2.py --stage index_f101   # official Food-101 archive
python scripts/build_dataset_v2.py --stage build
python scripts/run.py sweep
python scripts/select_lr.py
python scripts/run.py main ablation
python scripts/evaluate.py
python scripts/benchmark_inference.py
```

## Data availability

* **Food-101** is downloaded from its official source; the official train/test lists are used.
* **Turkish cuisine images**: the Kaggle *Turkish Food* dataset plus images collected by the author.
  The images are not redistributed. `data/manifest_v2.csv` lists every file with its source, MD5 and
  perceptual hash, so the exact split can be rebuilt from the original files.
