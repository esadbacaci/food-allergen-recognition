# Allergen database v2 — LLM drafting prompt (released verbatim)

**Model:** Claude Opus 5.5 (Anthropic), used through Claude Code on 2026-10-14.
**Output:** `llm_draft_v2.json` (raw, unedited). Every later change is made in separate, logged files
(`rule_check_log.csv`, `adjudication.csv`), never in the raw draft.

## Prompt

> You are drafting a dish-level allergen reference table for a food-recognition system. For each dish
> class below (identifier and English display name), write the ingredients of its *standard* recipe as
> commonly prepared (for Turkish dishes: as commonly prepared in Türkiye; for Food-101 dishes: as
> commonly served in Western restaurants), including typical accompaniments that are part of the dish
> as photographed (e.g. the bun of a hamburger, the bread of a döner wrap) but excluding optional side
> dishes.
>
> Then mark every allergen category of the taxonomy (`allergen_taxonomy_v2.csv`, codes GLU, MLK, EGG,
> PNT, NUT, SOY, FSH, CRU, SES, CEL, MUS, SUL, LUP, MOL, MAZ, AGS, RCE, SUN, PIN, PNA, KIW, GAR) that the
> dish contains, with one presence level:
> * **always** – part of the standard recipe; virtually every preparation contains it;
> * **often** – present in a substantial share of common recipes or regional variants;
> * **possible** – recipe-dependent, hidden or garnish ingredient, or typical cross-contact
>   (e.g. shared fryer), which a cautious user should be told about.
>
> Rules: list an allergen only if an ingredient you wrote explains it; do not mark refined oils or
> highly refined starches; a dish made of mammalian meat (beef, lamb, veal, pork, goat, offal) gets AGS;
> fish sauce, anchovy and Worcestershire sauce give FSH; soy sauce gives SOY and GLU (often);
> wine, vinegar-preserved and dried-fruit ingredients may give SUL. Return JSON only.

## How the draft is used

1. `scripts/build_allergen_db_v2.py` applies deterministic rule checks (ingredient lexicon → expected
   allergen) and logs every mismatch with its reason (`rule_check_log.csv`).
2. The draft is compared with the v1 database; every disagreement is listed in `v1_v2_diff.csv`.
3. Disagreements and rule-check flags are adjudicated by the authors (`adjudication.csv`), and a
   stratified subset is labelled blind by a dietitian (`expert_audit/`).
4. The released database (`allergens.csv`, `foods.csv`, `food_allergens.csv`, version 2.0) is generated
   from the draft + adjudication, never edited by hand.
