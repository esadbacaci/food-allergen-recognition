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
    """PIL-side transforms, ending in a uint8 CHW tensor (4x less inter-process traffic
    than float32). Scaling to [0, 1], ImageNet normalisation and, for the strong
    recipe, Random Erasing are applied on the GPU by `gpu_preprocess`; the result is
    numerically the same as ToTensor -> Normalize -> RandomErasing(p=0.25) on the CPU."""
    resize = int(round(img_size / 0.875))
    eval_t = T.Compose([
        T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.PILToTensor(),
    ])
    if aug == "strong":
        train_t = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.25, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.PILToTensor(),
        ])
    elif aug == "basic":  # the v1 augmentation, without the aspect-ratio distortion
        train_t = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.PILToTensor(),
        ])
    else:
        raise ValueError(aug)
    return train_t, eval_t


def random_erasing_(x, p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
    """Per-sample Random Erasing on a normalised batch (same sampling rule and fill
    value 0 as torchvision.transforms.RandomErasing)."""
    n, _, h, w = x.shape
    area = h * w
    log_ratio = torch.log(torch.tensor(ratio))
    for i in range(n):
        if torch.rand(1).item() >= p:
            continue
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            eh = int(round((erase_area * aspect) ** 0.5))
            ew = int(round((erase_area / aspect) ** 0.5))
            if eh < h and ew < w:
                top = torch.randint(0, h - eh + 1, (1,)).item()
                left = torch.randint(0, w - ew + 1, (1,)).item()
                x[i, :, top:top + eh, left:left + ew] = 0
                break
    return x


def gpu_preprocess(x, device, channels_last=False, erase=False):
    """uint8 batch -> normalised float batch on `device` (optionally Random Erasing)."""
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = x.to(device, non_blocking=True).float().div_(255).sub_(mean).div_(std)
    if erase:
        x = random_erasing_(x)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def _worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(batch_size, seed, img_size=224, aug="strong", num_workers=8,
                 eval_workers=4, eval_batch_size=128, image_dir="images"):
    """Every Windows worker process loads the full PyTorch/CUDA runtime (~1.5 GB commit),
    so only the training loader keeps persistent workers; evaluation workers are
    started per pass and released afterwards (36 persistent workers exhausted the
    page file on the 16 GB development machine)."""
    classes = read_class_map()
    class_ids = [c["class_id"] for c in classes]
    class_to_idx = {c: i for i, c in enumerate(class_ids)}
    rows = read_manifest()
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    train_t, eval_t = build_transforms(img_size, aug)
    root = paths.V2_DIR / image_dir

    g = torch.Generator()
    g.manual_seed(seed)

    def opts(workers, persistent):
        o = dict(num_workers=workers, pin_memory=True, worker_init_fn=_worker_init,
                 persistent_workers=persistent and workers > 0)
        if workers > 0:
            o["prefetch_factor"] = 4
        return o

    train_loader = DataLoader(ManifestDataset(by_split["train"], class_to_idx, train_t, root),
                              batch_size=batch_size, shuffle=True, drop_last=True,
                              generator=g, **opts(num_workers, True))
    val_loader = DataLoader(ManifestDataset(by_split["val"], class_to_idx, eval_t, root),
                            batch_size=eval_batch_size, shuffle=False, **opts(eval_workers, False))
    test_loader = DataLoader(ManifestDataset(by_split["test"], class_to_idx, eval_t, root),
                             batch_size=eval_batch_size, shuffle=False, **opts(eval_workers, False))

    counts = np.bincount([class_to_idx[r["cls"]] for r in by_split["train"]],
                         minlength=len(class_ids)).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1)
    w = w / w.sum() * len(class_ids)  # mean weight = 1 (same normalisation as v1)

    return dict(train=train_loader, val=val_loader, test=test_loader,
                class_ids=class_ids, class_weights=torch.tensor(w, dtype=torch.float32),
                val_rows=by_split["val"], test_rows=by_split["test"])
