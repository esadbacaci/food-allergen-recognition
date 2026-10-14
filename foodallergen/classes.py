# -*- coding: utf-8 -*-
"""Label system of the v2 dataset.

Class identifiers are lowercase ASCII with hyphens (as in v1 and in the allergen
database). Food-101 classes keep their official names (underscores -> hyphens).
Turkish-cuisine classes keep their Turkish identifier and get an English display
name, so the paper can report every class in one language.

Synonym merges (same dish under two names in v1): the images of the first class are
added to the second and the first class disappears. Three Turkish classes duplicated a
Food-101 dish; taze-fasulye / zeytinyagli-fasulye were the same Turkish dish (59 shared
near-duplicate images, checked visually). Generic/specific pairs (kebap / adana-kebap,
borek / su-boregi, enginar / canak-enginar) are different classes and are kept.
"""

# v1 folder -> v2 class (only for folders whose name changes)
SYNONYM_MERGES = {
    "patates-kizartmasi": "french-fries",
    "dondurma": "ice-cream",
    "omlet": "omelette",  # in v1 the omlet images sat inside eggs-benedict
    "zeytinyagli-fasulye": "taze-fasulye",
}

# Turkish identifier -> (English display name, Turkish display name)
TURKISH_CLASSES = {
    "adana-kebap": ("Adana kebab", "Adana kebap"),
    "armut": ("Pear", "Armut"),
    "asure": ("Ashure (Noah's pudding)", "Aşure"),
    "avokado": ("Avocado", "Avokado"),
    "ayran": ("Ayran (yogurt drink)", "Ayran"),
    "beyaz-lahana-sarmasi": ("Stuffed cabbage rolls", "Beyaz lahana sarması"),
    "biber-dolmasi": ("Stuffed peppers", "Biber dolması"),
    "borek": ("Börek (layered pastry)", "Börek"),
    "brokoli": ("Broccoli", "Brokoli"),
    "bruksel-lahanasi": ("Brussels sprouts dish", "Brüksel lahanası yemeği"),
    "bulgur-pilavi": ("Bulgur pilaf", "Bulgur pilavı"),
    "cacik": ("Cacık (yogurt with cucumber)", "Cacık"),
    "canak-enginar": ("Artichoke hearts in olive oil", "Çanak enginar"),
    "cay": ("Turkish tea", "Çay"),
    "cig-kofte": ("Çiğ köfte (bulgur patties)", "Çiğ köfte"),
    "cilek": ("Strawberry", "Çilek"),
    "cipura": ("Gilt-head sea bream", "Çipura"),
    "coban-salatasi": ("Shepherd's salad", "Çoban salatası"),
    "domates": ("Tomato", "Domates"),
    "domates-corbasi": ("Tomato soup", "Domates çorbası"),
    "doner": ("Döner kebab", "Döner"),
    "ekmek": ("Bread", "Ekmek"),
    "elma": ("Apple", "Elma"),
    "enginar": ("Artichoke dish", "Enginar yemeği"),
    "erik": ("Plum", "Erik"),
    "et-sote": ("Sautéed beef", "Et sote"),
    "gozleme": ("Gözleme (stuffed flatbread)", "Gözleme"),
    "hamsi": ("Fried anchovies", "Hamsi"),
    "haslanmis-yumurta": ("Boiled egg", "Haşlanmış yumurta"),
    "havuc": ("Carrot", "Havuç"),
    "hunkar-begendi": ("Hünkâr beğendi (lamb on eggplant purée)", "Hünkâr beğendi"),
    "icli-kofte": ("İçli köfte (stuffed bulgur balls)", "İçli köfte"),
    "incir": ("Fig", "İncir"),
    "iskender": ("İskender kebab", "İskender"),
    "ispanak": ("Spinach dish", "Ispanak yemeği"),
    "izmir-kofte": ("İzmir meatballs", "İzmir köfte"),
    "kabak-mucver": ("Zucchini fritters", "Kabak mücver"),
    "kalburabasti": ("Kalburabastı (syrup pastry)", "Kalburabastı"),
    "karnabahar": ("Cauliflower dish", "Karnabahar yemeği"),
    "karniyarik": ("Karnıyarık (stuffed eggplant)", "Karnıyarık"),
    "karpuz": ("Watermelon", "Karpuz"),
    "kavun": ("Melon", "Kavun"),
    "kayisi": ("Apricot", "Kayısı"),
    "kazandibi": ("Kazandibi (caramelised milk pudding)", "Kazandibi"),
    "kebap": ("Kebab (mixed)", "Kebap"),
    "kemal-pasa-tatlisi": ("Kemalpaşa dessert", "Kemalpaşa tatlısı"),
    "kiraz": ("Cherry", "Kiraz"),
    "kisir": ("Kısır (bulgur salad)", "Kısır"),
    "kivi": ("Kiwi", "Kivi"),
    "kiymali-borek": ("Börek with minced meat", "Kıymalı börek"),
    "kiymali-pide": ("Pide with minced meat", "Kıymalı pide"),
    "kofte": ("Meatballs (köfte)", "Köfte"),
    "kokorec": ("Kokoreç", "Kokoreç"),
    "kuru-fasulye": ("White bean stew", "Kuru fasulye"),
    "lahmacun": ("Lahmacun", "Lahmacun"),
    "levrek": ("European sea bass", "Levrek"),
    "lokma": ("Lokma (syrup doughnuts)", "Lokma"),
    "mango": ("Mango", "Mango"),
    "manti": ("Mantı (Turkish dumplings)", "Mantı"),
    "menemen": ("Menemen (eggs with tomato)", "Menemen"),
    "mercimek-corbasi": ("Lentil soup", "Mercimek çorbası"),
    "mercimek-koftesi": ("Lentil patties", "Mercimek köftesi"),
    "midye-dolma": ("Stuffed mussels", "Midye dolma"),
    "midye-tava": ("Fried mussels", "Midye tava"),
    "mumbar-dolmasi": ("Stuffed intestine (mumbar)", "Mumbar dolması"),
    "muz": ("Banana", "Muz"),
    "nar": ("Pomegranate", "Nar"),
    "patates-puresi": ("Mashed potatoes", "Patates püresi"),
    "patates-salatasi": ("Potato salad", "Patates salatası"),
    "patlican-kebabi": ("Eggplant kebab", "Patlıcan kebabı"),
    "peynirli-borek": ("Börek with cheese", "Peynirli börek"),
    "pirasa": ("Leeks in olive oil", "Zeytinyağlı pırasa"),
    "pirinc-pilavi": ("Rice pilaf", "Pirinç pilavı"),
    "portakal": ("Orange", "Portakal"),
    "sahlep": ("Salep (hot milk drink)", "Sahlep"),
    "salatalik": ("Cucumber", "Salatalık"),
    "salcali-makarna": ("Pasta with tomato paste", "Salçalı makarna"),
    "sandvic": ("Sandwich", "Sandviç"),
    "seftali": ("Peach", "Şeftali"),
    "sehriye-corbasi": ("Vermicelli soup", "Şehriye çorbası"),
    "simit": ("Simit (sesame bread ring)", "Simit"),
    "siyah-zeytin": ("Black olives", "Siyah zeytin"),
    "su-boregi": ("Su böreği (water börek)", "Su böreği"),
    "sucuklu-yumurta": ("Eggs with sucuk", "Sucuklu yumurta"),
    "sulu-bamya-yemegi": ("Okra stew", "Bamya yemeği"),
    "sulu-barbunya-yemegi": ("Borlotti bean stew", "Barbunya yemeği"),
    "sulu-bezelye-yemegi": ("Pea stew", "Bezelye yemeği"),
    "sulu-mercimek-yemegi": ("Green lentil stew", "Mercimek yemeği"),
    "sulu-nohut-yemegi": ("Chickpea stew", "Nohut yemeği"),
    "sulu-patates-yemegi": ("Potato stew", "Patates yemeği"),
    "sutlac": ("Rice pudding (sütlaç)", "Sütlaç"),
    "tantuni": ("Tantuni wrap", "Tantuni"),
    "tarhana-corbasi": ("Tarhana soup", "Tarhana çorbası"),
    "tas-kebabi": ("Tas kebab (meat stew)", "Tas kebabı"),
    "taze-fasulye": ("Green beans (incl. in olive oil)", "Taze fasulye (zeytinyağlı dahil)"),
    "tavuk-sote": ("Sautéed chicken", "Tavuk sote"),
    "tulumba-tatlisi": ("Tulumba (fried syrup dessert)", "Tulumba tatlısı"),
    "turk-kahvesi": ("Turkish coffee", "Türk kahvesi"),
    "tursu": ("Pickles (turşu)", "Turşu"),
    "uzum": ("Grapes", "Üzüm"),
    "yaprak-sarma": ("Stuffed vine leaves", "Yaprak sarma"),
    "yayla-corbasi": ("Yogurt soup (yayla)", "Yayla çorbası"),
    "yesil-zeytin": ("Green olives", "Yeşil zeytin"),
    "yogurt": ("Yogurt", "Yoğurt"),
    "yogurtlu-makarna": ("Pasta with yogurt", "Yoğurtlu makarna"),
}

# Single-ingredient raw fruit / vegetable classes, mostly stock photographs on plain
# backgrounds; reported separately (reviewer 3). Cooked vegetable dishes are not included.
RAW_PRODUCE = {  # checked visually on sample images (2026-10-14)
    "armut", "avokado", "brokoli", "cilek", "domates", "elma", "erik", "havuc", "incir",
    "karpuz", "kavun", "kayisi", "kiraz", "kivi", "mango", "muz", "nar", "portakal",
    "salatalik", "seftali", "uzum",
}


def food101_to_id(name):
    """Official Food-101 name (apple_pie) -> v2 class id (apple-pie)."""
    return name.replace("_", "-")


def food101_display(name):
    return name.replace("_", " ").capitalize()
