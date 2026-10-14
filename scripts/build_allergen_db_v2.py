# -*- coding: utf-8 -*-
"""Allergen database v2: rule check of the LLM draft, comparison with v1, adjudication sheet.

    python scripts/build_allergen_db_v2.py

Inputs : data/allergen_db_v2/llm_draft_v2.json (raw, never edited), data/allergen_taxonomy_v2.csv,
         data/allergen_db_v1/*.csv, data/allergen_db_v2/adjudication.csv (decisions, if present)
Outputs: data/allergen_db_v2/rule_check_log.csv  every rule flag with its reason
         data/allergen_db_v2/v1_v2_diff.csv      every class x allergen where v1 and v2 differ
         data/allergen_db_v2/adjudication.csv    template (keeps existing decisions)
         data/allergen_db_v2/summary.json

The rule check is two-sided, unlike the v1 filter (which could only remove entries):
  * missing     - an ingredient in the draft implies an allergen that is not marked
  * unsupported - an allergen is marked but no listed ingredient explains it
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.classes import SYNONYM_MERGES  # noqa: E402

D2 = paths.REPO_DATA / "allergen_db_v2"
D1 = paths.REPO_DATA / "allergen_db_v1"

# Ingredient lexicon: regex (word-boundary, case-insensitive) -> allergen code.
# Exceptions are applied before matching (e.g. "rice noodles" is not wheat).
EXCEPTIONS = [r"rice noodles?", r"glass noodles", r"corn flour", r"rice flour", r"almond flour",
              r"refined \w+ oil", r"cornstarch", r"corn tortilla chips", r"rice paper"]
LEXICON = {
    "GLU": [r"wheat", r"flour", r"bread", r"bun", r"pita", r"lava[sş]", r"yufka", r"phyllo", r"pasta",
            r"spaghetti", r"macaroni", r"noodles?", r"bulgur", r"semolina", r"croutons?", r"breadcrumbs",
            r"crackers?", r"dough", r"pastry", r"biscuit", r"vermicelli", r"orzo", r"barley", r"malt",
            r"soy sauce", r"gochujang", r"hoisin", r"ladyfingers", r"muffin", r"brioche", r"pide",
            r"roux", r"batter", r"tarhana", r"toast", r"sponge", r"shortcake", r"wafer cone", r"pancakes"],
    "MLK": [r"milk", r"butter", r"cream", r"cheese", r"yog(h)?urt", r"ricotta", r"mascarpone",
            r"mozzarella", r"parmesan", r"feta", r"gruy[eè]re", r"ka[sş]ar", r"b[eé]chamel", r"ghee",
            r"curds", r"pecorino", r"frosting", r"hollandaise", r"buttercream", r"ganache", r"ice cream"],
    "EGG": [r"eggs?", r"mayonnaise", r"aioli", r"hollandaise", r"tamago", r"ladyfingers", r"meringue"],
    "FSH": [r"fish", r"anchov(y|ies)", r"salmon", r"tuna", r"cod", r"haddock", r"sea bream",
            r"sea bass", r"bonito", r"dashi", r"worcestershire"],
    "CRU": [r"shrimp", r"prawns?", r"lobster", r"crab"],
    "MOL": [r"mussels?", r"clams?", r"oysters?", r"squid", r"calamari", r"octopus", r"snails?",
            r"scallops?"],
    "SES": [r"sesame", r"tahini"],
    "SOY": [r"soy", r"tofu", r"miso", r"edamame", r"soybeans", r"gochujang", r"hoisin"],
    "PNT": [r"peanuts?"],
    "NUT": [r"walnuts?", r"almonds?", r"hazelnuts?", r"pistachios?", r"cashews?", r"pecans?",
            r"macadamia"],
    "PIN": [r"pine nuts?"],
    "MUS": [r"mustard"],
    "CEL": [r"celery", r"celeriac"],
    "SUL": [r"wine", r"marsala", r"sherry", r"balsamic", r"dried (fruit|apricots?|figs?)", r"currants",
            r"raisins", r"vinegar"],
    "AGS": [r"beef", r"lamb", r"veal", r"pork", r"bacon", r"ham", r"sausage", r"sucuk", r"salami",
            r"past[ıi]rma", r"guanciale", r"pancetta", r"intestines?", r"tail fat", r"minced meat",
            r"meat", r"ribs", r"d[öo]ner", r"offal"],
    "RCE": [r"rice"],
    "MAZ": [r"corn", r"maize", r"grits", r"cornmeal"],
    "GAR": [r"garlic"],
    "KIW": [r"kiwi"],
}
V1_CODE = {1: "GLU", 2: "MLK", 3: "EGG", 4: "PNT", 5: "NUT", 6: "SOY", 7: "FSH", 8: "CRU", 9: "SES",
           10: "CEL", 11: "MUS", 12: "SUL", 13: "LUP", 14: "MOL", 15: "MAZ", 16: "AGS", 17: "RCE",
           18: "SUN", 19: "PIN", 20: "PNA", 21: "KIW", 22: "GAR", 23: "LTX"}
LEVEL_V1 = {"Yoğun": "high", "Orta": "medium", "Düşük": "low"}


def implied(ingredients):
    text = ingredients.lower()
    for ex in EXCEPTIONS:
        text = re.sub(ex, " ", text)
    hits = defaultdict(list)
    for code, pats in LEXICON.items():
        for p in pats:
            m = re.search(rf"\b{p}\b", text)
            if m:
                hits[code].append(m.group(0))
    return hits


def load_v1():
    foods = {r["YiyecekID"]: r["YiyecekAdi"] for r in csv.DictReader(open(D1 / "foods.csv", encoding="utf-8"))}
    v1 = defaultdict(dict)
    for r in csv.DictReader(open(D1 / "food_allergens.csv", encoding="utf-8")):
        v1[foods[r["YiyecekID"]]][V1_CODE[int(r["AlerjenID"])]] = LEVEL_V1.get(r["Miktar"], r["Miktar"])
    return v1


def main():
    draft = json.load(open(D2 / "llm_draft_v2.json", encoding="utf-8"))
    v1 = load_v1()
    v1_foods = {r["YiyecekAdi"] for r in csv.DictReader(open(D1 / "foods.csv", encoding="utf-8"))}

    flags, diff = [], []
    for e in draft:
        c, marked = e["class_id"], e["allergens"]
        hits = implied(e["ingredients"])
        for code, words in hits.items():
            if code not in marked:
                flags.append(dict(class_id=c, allergen=code, flag="missing",
                                  reason=f"ingredient(s) {', '.join(words)} imply {code}"))
        for code, level in marked.items():
            if code not in hits:
                flags.append(dict(class_id=c, allergen=code, flag="unsupported",
                                  reason=f"marked '{level}' but no listed ingredient implies {code}"))

        names = [c] + [a for a, b in SYNONYM_MERGES.items() if b == c]
        in_v1 = any(n in v1_foods for n in names)
        prof1 = {}
        for n in names:
            prof1.update(v1.get(n, {}))
        for code in sorted(set(prof1) | set(marked)):
            a, b = prof1.get(code), marked.get(code)
            v1_has, v2_has = a is not None, b in ("always", "often")
            if not in_v1:
                kind = "class_not_in_v1"
            elif v1_has and b is None:
                kind = "v1_only"
            elif v1_has and b == "possible":
                kind = "v1_present_v2_possible"
            elif not v1_has and v2_has:
                kind = "v2_only (v1 omission?)"
            elif not v1_has and b == "possible":
                kind = "v2_possible_only"
            else:
                continue
            diff.append(dict(class_id=c, allergen=code, v1_level=a or "", v2_level=b or "", kind=kind))

    with open(D2 / "rule_check_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "allergen", "flag", "reason"])
        w.writeheader()
        w.writerows(flags)
    with open(D2 / "v1_v2_diff.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "allergen", "v1_level", "v2_level", "kind"])
        w.writeheader()
        w.writerows(diff)

    # adjudication template: one row per rule flag or substantive v1/v2 disagreement
    adj_path = D2 / "adjudication.csv"
    existing = {}
    if adj_path.exists():
        for r in csv.DictReader(open(adj_path, encoding="utf-8")):
            existing[(r["class_id"], r["allergen"])] = r
    items = {}
    for d in diff:
        if d["kind"] in ("v1_only", "v2_only (v1 omission?)", "v1_present_v2_possible"):
            items[(d["class_id"], d["allergen"])] = dict(class_id=d["class_id"], allergen=d["allergen"],
                                                         v1_level=d["v1_level"], v2_level=d["v2_level"],
                                                         source=d["kind"])
    for fl in flags:
        key = (fl["class_id"], fl["allergen"])
        items.setdefault(key, dict(class_id=fl["class_id"], allergen=fl["allergen"], v1_level="",
                                   v2_level="", source=""))
        items[key]["source"] = (items[key]["source"] + "; " if items[key]["source"] else "") + f"rule:{fl['flag']}"
    with open(adj_path, "w", newline="", encoding="utf-8") as f:
        fields = ["class_id", "allergen", "v1_level", "v2_level", "source", "decision", "decided_by", "note"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(items):
            row = {**items[key], "decision": "", "decided_by": "", "note": ""}
            if key in existing:
                for k in ("decision", "decided_by", "note"):
                    row[k] = existing[key].get(k, "")
            w.writerow(row)

    summary = dict(draft_classes=len(draft),
                   draft_relations=Counter(lv for e in draft for lv in e["allergens"].values()),
                   rule_flags=Counter(f["flag"] for f in flags),
                   v1_v2_diff=Counter(d["kind"] for d in diff),
                   adjudication_items=len(items))
    (D2 / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
