# -*- coding: utf-8 -*-
"""Allergen database v2: proposed adjudication (explicit policy) and final release + MySQL migration.

    python scripts/finalize_allergen_db_v2.py propose    # fills empty decisions in adjudication.csv
    python scripts/finalize_allergen_db_v2.py build      # final CSVs + migration SQL

Proposed decisions are marked decided_by="proposed" and must be confirmed (decided_by changed to
the author's initials) before the release; `build` reports how many are still unconfirmed.

Policy (stated in the article, Section 2.2):
  * v2 marks always/often, v1 lacks it            -> accept v2 (omission in v1; ingredient-supported)
  * v1 lists it, v2 lacks it or says possible      -> keep at least 'possible' (conservative rule)
  * latex (not a food)                             -> remove
  * rule 'missing' from 'meat' in crab/lobster meat -> reject (lexicon false positive)
  * rule 'missing' from vinegar / currants (SUL, RCE) -> add as 'possible'
  * rule 'unsupported' on a 'possible' mark        -> keep 'possible' (hidden ingredient / cross-contact)
  * rule 'unsupported' on always/often because the lexicon exceptions hid the ingredient
    (corn flour, rice flour, rice noodles) or a generic word (nuts, garlic in sauces) -> accept v2
Explicit overrides are listed in OVERRIDES with their reason.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.classes import SYNONYM_MERGES  # noqa: E402

D2 = paths.REPO_DATA / "allergen_db_v2"
ADJ = D2 / "adjudication.csv"
LEVEL_TR = {"always": "Her zaman", "often": "Sıklıkla", "possible": "Olabilir"}
VERSION = "2.0"

OVERRIDES = {  # (class, code): (decision, note)
    ("fried-calamari", "EGG"): ("set:often", "egg is common in calamari batter; v1 medium"),
    ("crab-cakes", "AGS"): ("reject", "'crab meat' is not mammalian meat (lexicon false positive)"),
    ("lobster-roll-sandwich", "AGS"): ("reject", "'lobster meat' is not mammalian meat (lexicon false positive)"),
}


def propose():
    rows = list(csv.DictReader(open(ADJ, encoding="utf-8")))
    n = 0
    for r in rows:
        if r["decision"]:
            continue
        key, src, v1, v2 = (r["class_id"], r["allergen"]), r["source"], r["v1_level"], r["v2_level"]
        if key in OVERRIDES:
            d, note = OVERRIDES[key]
        elif r["allergen"] == "LTX":
            d, note = "remove", "latex is not a food; moved to user information text"
        elif "v2_only" in src:
            d, note = "accept_v2", "missing in v1; implied by the standard recipe"
        elif "v1_only" in src or "v1_present_v2_possible" in src:
            d, note = "set:possible", "reported by v1; kept at least as 'possible' (conservative rule)"
        elif "rule:missing" in src:
            d, note = "set:possible", "implied by vinegar / dried fruit / rice vinegar; recipe-dependent"
        elif "rule:unsupported" in src and v2 in ("", "possible"):
            d, note = "accept_v2", "hidden ingredient or cross-contact; kept as 'possible'"
        else:
            d, note = "accept_v2", "ingredient hidden by lexicon exceptions or generic wording"
        r["decision"], r["decided_by"], r["note"] = d, "proposed", note
        n += 1
    with open(ADJ, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"proposed {n} decisions:", dict(Counter(r["decision"] for r in rows)))


def build():
    tax = list(csv.DictReader(open(paths.REPO_DATA / "allergen_taxonomy_v2.csv", encoding="utf-8")))
    tax = [t for t in tax if t["v2_status"] != "removed"]
    code_id = {t["code"]: int(t["db_id"]) for t in tax}
    classes = sorted(csv.DictReader(open(paths.CLASS_MAP_V2, encoding="utf-8")), key=lambda r: int(r["index"]))
    draft = {e["class_id"]: e for e in json.load(open(D2 / "llm_draft_v2.json", encoding="utf-8"))}
    # merged classes: the draft of the merged-away class is folded in at its lowest level
    rel = defaultdict(dict)  # class -> code -> (level, source)
    order = {"always": 3, "often": 2, "possible": 1}
    for c, e in draft.items():
        tgt = SYNONYM_MERGES.get(c, c)
        for code, lv in e["allergens"].items():
            lv = lv if tgt == c else "possible"
            if code not in rel[tgt] or order[lv] > order[rel[tgt][code][0]]:
                rel[tgt][code] = (lv, "draft")
    unconfirmed = 0
    for r in csv.DictReader(open(ADJ, encoding="utf-8")):
        c, code, d = SYNONYM_MERGES.get(r["class_id"], r["class_id"]), r["allergen"], r["decision"]
        unconfirmed += r["decided_by"] in ("", "proposed")
        if d in ("remove", "reject"):
            rel[c].pop(code, None)
        elif d.startswith("set:"):
            rel[c][code] = (d.split(":")[1], "adjudication")
        elif d == "accept_v2" and code in rel[c]:
            rel[c][code] = (rel[c][code][0], "draft+adjudication")
    expert = D2 / "expert_audit" / "expert_decisions.csv"  # optional: applied after the audit
    if expert.exists():
        for r in csv.DictReader(open(expert, encoding="utf-8")):
            rel[r["class_id"]][r["allergen"]] = (r["level"], "expert")

    class_ids = [c["class_id"] for c in classes]
    missing = [c for c in class_ids if c not in draft]
    D2.mkdir(exist_ok=True)
    out_rel = []
    for i, c in enumerate(class_ids, 1):
        for code, (lv, src) in sorted(rel[c].items()):
            if code in code_id:
                out_rel.append(dict(YiyecekID=i, AlerjenID=code_id[code], Miktar=LEVEL_TR[lv], Kaynak=src,
                                    Surum=VERSION, class_id=c, code=code))
    write(D2 / "food_allergens.csv", out_rel)
    write(D2 / "foods.csv", [dict(YiyecekID=i, YiyecekAdi=c["class_id"], GorunenAdEN=c["display_en"],
                                  GorunenAdTR=c["display_tr"]) for i, c in enumerate(classes, 1)])
    write(D2 / "allergens.csv", [dict(AlerjenID=t["db_id"], Kod=t["code"], AlerjenAdi=t["name_tr"],
                                      AlerjenAdiEN=t["name_en"], Grup=t["group"], YasalDayanak=t["legal_basis"])
                                 for t in tax])
    sql(D2 / "migration_v1_to_v2.sql", tax, classes, out_rel)
    summary = dict(version=VERSION, classes=len(class_ids), relations=len(out_rel),
                   by_level=Counter(r["Miktar"] for r in out_rel), by_source=Counter(r["Kaynak"] for r in out_rel),
                   regulated_relations=sum(1 for r in out_rel if r["AlerjenID"] <= 14),
                   classes_without_draft=missing, unconfirmed_decisions=unconfirmed)
    (D2 / "release_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def q(s):
    return "'" + str(s).replace("\\", "\\\\").replace("'", "''") + "'"


GROUP_TR = {"A_regulated": "Yönetmelikle düzenlenen alerjen", "B_additional_food": "Ek gıda kategorisi",
            "C_syndrome": "Alfa-gal sendromu"}


def sql(path, tax, classes, rel):
    """MySQL 8.0: run ONCE on the v1 database (no IF NOT EXISTS for columns in MySQL). DDL commits
    implicitly in MySQL, so backups and schema changes come first and the data replacement is the
    only transactional part."""
    L = ["-- Allergen database v1 -> v2 migration (generated by 02_kod/scripts/finalize_allergen_db_v2.py).",
         "-- Run once on the v1 database (MySQL 8.0). Rollback: restore the *_v1 backup tables.",
         "SET NAMES utf8mb4;",
         "-- 1) backups of v1",
         "CREATE TABLE alerjenler_v1 AS SELECT * FROM alerjenler;",
         "CREATE TABLE yiyecekler_v1 AS SELECT * FROM yiyecekler;",
         "CREATE TABLE yiyecekalerjenleri_v1 AS SELECT * FROM yiyecekalerjenleri;",
         "-- 2) schema additions (the application ignores columns it does not map)",
         "ALTER TABLE alerjenler ADD COLUMN Kod VARCHAR(8), ADD COLUMN AlerjenAdiEN VARCHAR(100),"
         " ADD COLUMN Grup VARCHAR(40), ADD COLUMN YasalDayanak VARCHAR(255);",
         "ALTER TABLE yiyecekler ADD COLUMN GorunenAdEN VARCHAR(255), ADD COLUMN GorunenAdTR VARCHAR(255);",
         "ALTER TABLE yiyecekalerjenleri ADD COLUMN Kaynak VARCHAR(40), ADD COLUMN Surum VARCHAR(10);",
         "CREATE TABLE versiyonlar (Surum VARCHAR(10) PRIMARY KEY, Tarih DATE, Aciklama VARCHAR(500));",
         "-- 3) data replacement (atomic)",
         "START TRANSACTION;",
         "DELETE FROM yiyecekalerjenleri;", "DELETE FROM yiyecekler;", "DELETE FROM alerjenler;",
         f"INSERT INTO versiyonlar VALUES ({q(VERSION)}, CURDATE(), 'v2: 206 classes, regulated taxonomy, presence levels, adjudicated LLM draft');"]
    L.append("INSERT INTO alerjenler (AlerjenID, AlerjenAdi, AlerjenKategori, Kod, AlerjenAdiEN, Grup, YasalDayanak) VALUES")
    L.append(",\n".join(f"({t['db_id']}, {q(t['name_tr'])}, {q(GROUP_TR[t['group']])}, {q(t['code'])}, "
                        f"{q(t['name_en'])}, {q(t['group'])}, {q(t['legal_basis'])})" for t in tax) + ";")
    L.append("INSERT INTO yiyecekler (YiyecekID, YiyecekAdi, Kategori, GorunenAdEN, GorunenAdTR) VALUES")
    L.append(",\n".join(f"({i}, {q(c['class_id'])}, {q(c['group'])}, {q(c['display_en'])}, {q(c['display_tr'])})"
                        for i, c in enumerate(classes, 1)) + ";")
    L.append("INSERT INTO yiyecekalerjenleri (YiyecekID, AlerjenID, Miktar, Kaynak, Surum) VALUES")
    L.append(",\n".join(f"({r['YiyecekID']}, {r['AlerjenID']}, {q(r['Miktar'])}, {q(r['Kaynak'])}, {q(r['Surum'])})"
                        for r in rel) + ";")
    L.append("COMMIT;")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"propose": propose, "build": build}[sys.argv[1]]()
