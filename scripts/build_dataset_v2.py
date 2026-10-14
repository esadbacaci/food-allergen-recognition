# -*- coding: utf-8 -*-
"""Build the v2 dataset (fixes the label errors of v1, deduplicates, re-splits).

Stages (each one caches its output, so they can be run separately):
  index_v1    MD5 + pHash + size of every image of the v1 dataset (the one behind the
              submitted article).
  index_f101  Extract the official Food-101 archive (if needed) and index it.
  build       Provenance, label repair, deduplication, splits, resized image copies,
              manifest and report.

    python scripts/build_dataset_v2.py --stage index_v1
    python scripts/build_dataset_v2.py --stage index_f101
    python scripts/build_dataset_v2.py --stage build

Design (see data/dataset_report_v2.json for the resulting numbers):
  * Food-101 part = the official archive (all 101 classes, official train/test lists).
    v1 images are matched to it by MD5, which also reveals v1 label errors.
  * Turkish part = every v1 image that is not a Food-101 image. Source is read from the
    file name: 1-4 digit numeric names come from the Kaggle Turkish Food dataset, named
    files (<class>NNNN.jpg) were collected by the author.
  * Synonym classes are merged into the Food-101 class (foodallergen.classes).
  * Near-duplicates (pHash Hamming <= 5) and exact duplicates (same MD5) are grouped:
      - a Turkish image that duplicates a Food-101 image is dropped;
      - Turkish-only groups keep one image (largest resolution); if the group spans
        several classes the whole group is dropped (ambiguous label);
      - Food-101 train images that duplicate a Food-101 test image are dropped
        (official test set itself is never changed).
  * Splits: Food-101 test = official test; official train -> 90 % train / 10 % val
    (stratified). Turkish images -> 70/15/15 stratified. Seed 42.
  * Images are stored resized (shorter side 320 px, JPEG q95) for fast training.
"""
import argparse
import csv
import json
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodallergen import paths  # noqa: E402
from foodallergen.classes import (  # noqa: E402
    RAW_PRODUCE, SYNONYM_MERGES, TURKISH_CLASSES, food101_display, food101_to_id)
from foodallergen.hashing import (  # noqa: E402
    UnionFind, exact_duplicate_pairs, file_signature, near_duplicate_pairs)

SEED = 42
PHASH_THRESHOLD = 5
SHORT_SIDE = 320
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".jfif"}
WORKERS = 12

V1_INDEX = paths.V2_CACHE / "v1_index.csv"
F101_INDEX = paths.V2_CACHE / "f101_index.csv"
INDEX_FIELDS = ["path", "md5", "phash", "width", "height", "ok"]


# ----------------------------------------------------------------------------- index
def _sig(p):
    return (p, *file_signature(p))


def write_index(files, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(WORKERS) as pool:
        for k, r in enumerate(pool.imap(_sig, files, chunksize=64)):
            rows.append(r)
            if (k + 1) % 10000 == 0:
                print(f"  {k + 1}/{len(files)}", flush=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(INDEX_FIELDS)
        for p, md5, ph, wd, ht, ok in rows:
            w.writerow([p, md5, ph, wd, ht, int(ok)])
    bad = sum(1 for r in rows if not r[-1])
    print(f"  -> {out_csv} ({len(rows)} files, unreadable: {bad})")


def read_index(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def list_images(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() in IMG_EXT:
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def stage_index_v1():
    files = list_images(paths.V1_DATASET)
    print(f"v1: {len(files)} image files under {paths.V1_DATASET}")
    write_index(files, V1_INDEX)


def stage_index_f101():
    if not (paths.FOOD101_DIR / "meta" / "train.txt").exists():
        print(f"Extracting {paths.FOOD101_TAR} ...", flush=True)
        with tarfile.open(paths.FOOD101_TAR, "r:gz") as tar:
            tar.extractall(paths.SOURCE)  # archive root folder is 'food-101'
    files = list_images(paths.FOOD101_DIR / "images")
    print(f"Food-101: {len(files)} image files")
    write_index(files, F101_INDEX)


# ----------------------------------------------------------------------------- build
def load_food101_meta():
    meta = paths.FOOD101_DIR / "meta"
    classes = (meta / "classes.txt").read_text(encoding="utf-8").split()
    split_of = {}
    for split in ("train", "test"):
        for line in (meta / f"{split}.txt").read_text(encoding="utf-8").split():
            split_of[line.strip()] = split  # 'apple_pie/1005649' -> 'train'
    return classes, split_of


def tr_source(filename):
    stem = Path(filename).stem
    if re.fullmatch(r"\d{1,4}", stem):
        return "tr_kaggle"
    return "tr_collected"


def stratified_split(items_by_class, fractions, rng):
    """items_by_class: {cls: [item, ...]} -> {item: split} with per-class shuffling."""
    names = list(fractions)
    out = {}
    for cls in sorted(items_by_class):
        items = sorted(items_by_class[cls])
        rng.shuffle(items)
        n = len(items)
        cuts, acc = [], 0.0
        for name in names[:-1]:
            acc += fractions[name]
            cuts.append(int(round(acc * n)))
        bounds = [0] + cuts + [n]
        for k, name in enumerate(names):
            for it in items[bounds[k]:bounds[k + 1]]:
                out[it] = name
    return out


def _resize_one(args):
    src, dst = args
    from PIL import Image, ImageOps
    if os.path.exists(dst):
        return True
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            s = SHORT_SIDE / min(w, h)
            if s < 1:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            im.save(dst, "JPEG", quality=95)
        return True
    except Exception:
        return False


def stage_build():
    f101_classes, f101_split = load_food101_meta()
    f101_rows = read_index(F101_INDEX)
    v1_rows = read_index(V1_INDEX)
    report = {"seed": SEED, "phash_threshold": PHASH_THRESHOLD, "short_side": SHORT_SIDE}

    # ---- pool: official Food-101 images
    pool = []  # dicts: uid, cls, source, src, md5, phash, w, h, f101_split
    md5_to_f101 = {}
    for r in f101_rows:
        p = Path(r["path"])
        key = f"{p.parent.name}/{p.stem}"
        assert key in f101_split, key
        it = dict(uid=f"f101_{p.stem}", cls=food101_to_id(p.parent.name), source="food101",
                  src=r["path"], md5=r["md5"], phash=int(r["phash"]), w=int(r["width"]),
                  h=int(r["height"]), f101_split=f101_split[key], v1_folder="")
        md5_to_f101[r["md5"]] = it
        pool.append(it)

    # ---- v1 images: provenance + label audit
    audit = Counter()          # (v1 folder, official class) for Food-101 images
    v1_f101_found = set()
    unreadable_v1 = 0
    for r in v1_rows:
        p = Path(r["path"])
        folder = p.parent.name
        if r["ok"] != "1":
            unreadable_v1 += 1
            continue
        hit = md5_to_f101.get(r["md5"])
        if hit is not None:
            hit["v1_folder"] = folder
            v1_f101_found.add(hit["uid"])
            audit[(folder, hit["cls"])] += 1
            continue
        cls = SYNONYM_MERGES.get(folder, folder)
        if folder == "eggs-benedict" and p.stem.lower().startswith("omlet"):
            cls = "omelette"
        src_kind = tr_source(p.name)
        pool.append(dict(uid=f"{'kg' if src_kind == 'tr_kaggle' else 'col'}_{folder}_{p.stem}",
                         cls=cls, source=src_kind, src=r["path"], md5=r["md5"],
                         phash=int(r["phash"]), w=int(r["width"]), h=int(r["height"]),
                         f101_split="", v1_folder=folder))

    seen_uid = Counter()
    for it in pool:  # same stem with different extensions -> keep uids unique
        seen_uid[it["uid"]] += 1
        if seen_uid[it["uid"]] > 1:
            it["uid"] += "_" + Path(it["src"]).suffix.lstrip(".").lower()
    assert len({it["uid"] for it in pool}) == len(pool), "uid collision"

    v2_f101_ids = {food101_to_id(c) for c in f101_classes}
    tr_classes = sorted({it["cls"] for it in pool if it["source"] != "food101"} - v2_f101_ids)
    unknown = [c for c in tr_classes if c not in TURKISH_CLASSES]
    assert not unknown, f"classes without display name: {unknown}"

    mislabeled = {f"{a} -> {b}": n for (a, b), n in sorted(audit.items()) if a != b}
    report["v1"] = {
        "images": len(v1_rows),
        "unreadable": unreadable_v1,
        "food101_images_found_by_md5": len(v1_f101_found),
        "food101_images_in_wrong_folder": mislabeled,
        "food101_classes_absent_in_v1": sorted(
            {food101_to_id(c) for c in f101_classes} - {b for (_, b) in audit}),
        "food101_images_missing_in_v1": len(f101_rows) - len(v1_f101_found),
        "turkish_images": sum(1 for it in pool if it["source"] != "food101"),
    }

    # ---- duplicates
    hashes = np.array([it["phash"] for it in pool], dtype=np.uint64)
    pairs = near_duplicate_pairs(hashes, PHASH_THRESHOLD)
    for k, d in exact_duplicate_pairs([it["md5"] for it in pool]).items():
        pairs[k] = 0
    uf = UnionFind(len(pool))
    for i, j in pairs:
        uf.union(i, j)
    groups = defaultdict(list)
    for i in range(len(pool)):
        groups[uf.find(i)].append(i)

    drop = {}  # index -> reason
    cross_class_groups = Counter()
    for members in groups.values():
        if len(members) < 2:
            continue
        f101 = [i for i in members if pool[i]["source"] == "food101"]
        tr = [i for i in members if pool[i]["source"] != "food101"]
        if f101:
            for i in tr:
                drop[i] = "duplicate_of_food101"
            if any(pool[i]["f101_split"] == "test" for i in f101):
                for i in f101:
                    if pool[i]["f101_split"] == "train":
                        drop[i] = "food101_train_duplicate_of_test"
            continue
        classes_in_group = {pool[i]["cls"] for i in tr}
        if len(classes_in_group) > 1:
            cross_class_groups[" | ".join(sorted(classes_in_group))] += 1
            for i in tr:
                drop[i] = "cross_class_duplicate"
            continue
        keep = max(tr, key=lambda i: (pool[i]["w"] * pool[i]["h"], pool[i]["uid"]))
        for i in tr:
            if i != keep:
                drop[i] = "duplicate_within_class"
    unreadable_pool = [i for i in range(len(pool)) if not pool[i]["md5"]]
    for i in unreadable_pool:
        drop[i] = "unreadable"

    pair_kinds = Counter()
    for (i, j), d in pairs.items():
        a, b = pool[i], pool[j]
        kind = "f101-f101" if a["source"] == b["source"] == "food101" else (
            "tr-tr" if "food101" not in (a["source"], b["source"]) else "f101-tr")
        pair_kinds[f"{kind}{'' if a['cls'] == b['cls'] else ' (cross-class)'}"] += 1
    report["deduplication"] = {
        "duplicate_pairs": len(pairs),
        "exact_pairs": sum(1 for d in pairs.values() if d == 0),
        "pairs_by_kind": dict(pair_kinds),
        "cross_class_groups": dict(cross_class_groups.most_common()),
        "dropped_by_reason": dict(Counter(drop.values())),
    }

    # ---- splits
    rng = np.random.default_rng(SEED)
    kept = [i for i in range(len(pool)) if i not in drop]
    f101_train = defaultdict(list)
    tr_items = defaultdict(list)
    split = {}
    for i in kept:
        it = pool[i]
        if it["source"] == "food101":
            if it["f101_split"] == "test":
                split[i] = "test"
            else:
                f101_train[it["cls"]].append(i)
        else:
            tr_items[it["cls"]].append(i)
    split.update(stratified_split(f101_train, {"train": 0.9, "val": 0.1}, rng))
    split.update(stratified_split(tr_items, {"train": 0.70, "val": 0.15, "test": 0.15}, rng))
    # Food-101 duplicate groups inside the official train list stay in one partition.
    moved_to_train = 0
    for members in groups.values():
        in_both = {split.get(i) for i in members if i in split} >= {"train", "val"}
        if in_both:
            for i in members:
                if split.get(i) == "val":
                    split[i] = "train"
                    moved_to_train += 1
    report["deduplication"]["food101_val_images_moved_to_train_group_aware"] = moved_to_train

    # ---- verification: no near-duplicate across splits among kept images
    kept_set = set(kept)
    leaks = Counter()
    for (i, j) in pairs:
        if i in kept_set and j in kept_set and split[i] != split[j]:
            leaks["-".join(sorted((split[i], split[j])))] += 1
    report["verification_cross_split_pairs_after_dedup"] = dict(leaks)

    # ---- write resized images
    jobs, rows = [], []
    for i in kept:
        it = pool[i]
        rel = f"{it['cls']}/{it['uid']}.jpg"
        jobs.append((it["src"], str(paths.V2_IMAGES / rel)))
        rows.append(dict(uid=it["uid"], cls=it["cls"], source=it["source"], split=split[i],
                         path=rel, md5=it["md5"], phash=f"{it['phash']:016x}",
                         orig_width=it["w"], orig_height=it["h"],
                         orig_name=f"{it['v1_folder'] or Path(it['src']).parent.name}/{Path(it['src']).name}"))
    print(f"Writing {len(jobs)} resized images to {paths.V2_IMAGES} ...", flush=True)
    with Pool(WORKERS) as p:
        ok = p.map(_resize_one, jobs, chunksize=64)
    failed = [jobs[k][0] for k, good in enumerate(ok) if not good]
    report["resize_failures"] = failed

    rows.sort(key=lambda r: (r["cls"], r["uid"]))
    paths.REPO_DATA.mkdir(parents=True, exist_ok=True)
    with open(paths.MANIFEST_V2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- class map + statistics
    classes = sorted({r["cls"] for r in rows})
    per = defaultdict(Counter)
    for r in rows:
        per[r["cls"]][r["split"]] += 1
        per[r["cls"]][r["source"]] += 1
    f101_display = {food101_to_id(c): food101_display(c) for c in f101_classes}
    with open(paths.CLASS_MAP_V2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "class_id", "display_en", "display_tr", "group", "raw_produce",
                    "merged_from", "n_train", "n_val", "n_test", "n_food101", "n_tr_kaggle",
                    "n_tr_collected"])
        merged_into = defaultdict(list)
        for a, b in SYNONYM_MERGES.items():
            merged_into[b].append(a)
        for k, c in enumerate(classes):
            if c in f101_display:
                en, tr_name, group = f101_display[c], "", "food101"
            else:
                en, tr_name = TURKISH_CLASSES[c]
                group = "turkish"
            w.writerow([k, c, en, tr_name, group, int(c in RAW_PRODUCE),
                        "+".join(merged_into.get(c, [])), per[c]["train"], per[c]["val"],
                        per[c]["test"], per[c]["food101"], per[c]["tr_kaggle"],
                        per[c]["tr_collected"]])

    sizes = np.array([sum(per[c][s] for s in ("train", "val", "test")) for c in classes])
    report["v2"] = {
        "classes": len(classes),
        "food101_classes": sum(1 for c in classes if c in f101_display),
        "turkish_classes": sum(1 for c in classes if c not in f101_display),
        "images": len(rows),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_source_and_split": {f"{s}/{t}": n for (s, t), n in sorted(
            Counter((r["source"], r["split"]) for r in rows).items())},
        "images_per_class": {"min": int(sizes.min()), "max": int(sizes.max()),
                             "mean": round(float(sizes.mean()), 1),
                             "median": float(np.median(sizes))},
        "smallest_classes": sorted(((int(n), c) for n, c in zip(sizes, classes)))[:10],
    }
    with open(paths.DATASET_REPORT_V2, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("v1", "deduplication", "v2")},
                     ensure_ascii=False, indent=2))
    print("verification:", report["verification_cross_split_pairs_after_dedup"],
          "| resize failures:", len(failed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["index_v1", "index_f101", "build"])
    stage = ap.parse_args().stage
    {"index_v1": stage_index_v1, "index_f101": stage_index_f101, "build": stage_build}[stage]()
