"""Single source of truth for all paths.

Every path is derived from the project root, which defaults to the parent of this
repository (``<root>/02_kod``). Set the ``FOOD_PROJECT_ROOT`` environment variable to
run the code from a different location; nothing else has to change.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("FOOD_PROJECT_ROOT", REPO.parent)).resolve()

# --- data (kept outside the repository) ---
DATA = PROJECT / "01_veri"
SOURCE = DATA / "kaynak"
FOOD101_TAR = SOURCE / "food-101.tar.gz"
FOOD101_DIR = SOURCE / "food-101"            # extracted official archive
V1_DATASET = DATA / "v1_gonderilen" / "food-dataset"
V2_DIR = DATA / "v2"
V2_IMAGES = V2_DIR / "images"
V2_CACHE = V2_DIR / "cache"                   # hashes etc., recomputable

# --- versioned, shareable artefacts (inside the repository) ---
REPO_DATA = REPO / "data"
MANIFEST_V2 = REPO_DATA / "manifest_v2.csv"
CLASS_MAP_V2 = REPO_DATA / "class_map_v2.csv"
DATASET_REPORT_V2 = REPO_DATA / "dataset_report_v2.json"

# --- experiment outputs ---
RUNS = PROJECT / "03_deneyler"
ALLERGEN_DB = PROJECT / "04_alerjen_db"
