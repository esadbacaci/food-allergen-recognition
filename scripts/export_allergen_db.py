# -*- coding: utf-8 -*-
"""Export the allergen database (MySQL dump) to versioned CSV files.

    python scripts/export_allergen_db.py [path/to/food.sql]

Writes data/allergen_db_v1/{allergens,foods,food_allergens}.csv and
data/allergen_db_v1/class_coverage.csv (which v2 classes have a database entry).
The live database (phpMyAdmin, 2026-10-14) had the same row counts as this dump:
23 allergens, 206 foods, 334 food-allergen relations.
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.classes import SYNONYM_MERGES  # noqa: E402

SQL = Path(sys.argv[1]) if len(sys.argv) > 1 else paths.ALLERGEN_DB / "v1" / "food.sql"
OUT = paths.REPO_DATA / "allergen_db_v1"


def rows_of(sql, table):
    m = re.search(rf"insert\s+into\s+`{table}`\s*\(([^)]*)\)\s*values\s*(.*?);\s*$", sql,
                  re.S | re.I | re.M)
    cols = [c.strip(" `") for c in m.group(1).split(",")]
    out = []
    for tup in re.findall(r"\((.*?)\)(?:,|$)", m.group(2), re.S):
        vals = [v.strip() for v in re.findall(r"'(?:[^'\\]|\\.)*'|NULL|-?\d+", tup)]
        vals = [None if v == "NULL" else v[1:-1] if v.startswith("'") else int(v) for v in vals]
        out.append(dict(zip(cols, vals)))
    return out


def main():
    sql = SQL.read_text(encoding="utf-8")
    allergens = rows_of(sql, "alerjenler")
    foods = rows_of(sql, "yiyecekler")
    rel = rows_of(sql, "yiyecekalerjenleri")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [("allergens", allergens), ("foods", foods), ("food_allergens", rel)]:
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"allergens {len(allergens)} | foods {len(foods)} | relations {len(rel)}")

    food_names = {f["YiyecekAdi"] for f in foods}
    if paths.CLASS_MAP_V2.exists():
        with open(paths.CLASS_MAP_V2, encoding="utf-8") as f:
            classes = [r["class_id"] for r in csv.DictReader(f)]
        reverse = {}
        for a, b in SYNONYM_MERGES.items():
            reverse.setdefault(b, []).append(a)
        cov = []
        for c in classes:
            names = [n for n in [c, *reverse.get(c, [])] if n in food_names]
            cov.append(dict(class_id=c, db_entries="+".join(names), covered=int(bool(names))))
        with open(OUT / "class_coverage.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cov[0]))
            w.writeheader()
            w.writerows(cov)
        missing = [r["class_id"] for r in cov if not r["covered"]]
        extra = sorted(food_names - {n for r in cov for n in r["db_entries"].split("+") if n})
        print("v2 classes without a DB entry:", missing)
        print("DB foods that are not v2 classes:", extra)


if __name__ == "__main__":
    main()
