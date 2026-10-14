"""Manifest-based dataset, transforms and seeded data loaders."""
import csv
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from . import paths

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_class_map(path=None):
    with open(path or paths.CLASS_MAP_V2, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["index"]))
    return rows


def read_manifest(path=None):
    with open(path or paths.MANIFEST_V2, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ManifestDataset(Dataset):
    def __init__(self, rows, class_to_idx, transform, root=None):
        self.root = root or paths.V2_IMAGES
        self.items = [(r["path"], class_to_idx[r["cls"]]) for r in rows]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y = self.items[i]
        with Image.open(self.root / rel) as im:
            x = self.transform(im.convert("RGB"))
        return x, y


def build_transforms(img_size=224, aug="strong"):
    resize = int(round(img_size / 0.875))
    eval_t = T.Compose([
        T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    if aug == "strong":
        train_t = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.25, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.25),
        ])
    elif aug == "basic":  # the v1 augmentation, without the aspect-ratio distortion
        train_t = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    else:
        raise ValueError(aug)
    return train_t, eval_t


def _worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(batch_size, seed, img_size=224, aug="strong", num_workers=12,
                 eval_batch_size=128):
    classes = read_class_map()
    class_ids = [c["class_id"] for c in classes]
    class_to_idx = {c: i for i, c in enumerate(class_ids)}
    rows = read_manifest()
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    train_t, eval_t = build_transforms(img_size, aug)

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
                  worker_init_fn=_worker_init)
    if num_workers > 0:
        common["prefetch_factor"] = 4
    train_loader = DataLoader(ManifestDataset(by_split["train"], class_to_idx, train_t),
                              batch_size=batch_size, shuffle=True, drop_last=True,
                              generator=g, **common)
    val_loader = DataLoader(ManifestDataset(by_split["val"], class_to_idx, eval_t),
                            batch_size=eval_batch_size, shuffle=False, **common)
    test_loader = DataLoader(ManifestDataset(by_split["test"], class_to_idx, eval_t),
                             batch_size=eval_batch_size, shuffle=False, **common)

    counts = np.bincount([class_to_idx[r["cls"]] for r in by_split["train"]],
                         minlength=len(class_ids)).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1)
    w = w / w.sum() * len(class_ids)  # mean weight = 1 (same normalisation as v1)

    return dict(train=train_loader, val=val_loader, test=test_loader,
                class_ids=class_ids, class_weights=torch.tensor(w, dtype=torch.float32),
                val_rows=by_split["val"], test_rows=by_split["test"])
