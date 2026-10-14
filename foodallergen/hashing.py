"""Exact (MD5) and perceptual (64-bit DCT pHash) hashing plus fast near-duplicate search.

The pHash implementation is identical to the one used for the v1 duplicate scan
(07_v1_deneyler/revizyon/07_duplicate_scan), so v1 and v2 numbers are comparable.
"""
import hashlib
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.fftpack import dct

BAD_HASH = np.uint64(0xFFFFFFFFFFFFFFFF)
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash64_from_image(im):
    im.draft("L", (32, 32))  # decode JPEGs at low resolution: much faster
    im = im.convert("L").resize((32, 32), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = d.flatten()
    med = np.median(flat[1:])  # exclude the DC component from the median
    h = 0
    for b in flat > med:
        h = (h << 1) | int(b)
    return np.uint64(h)


def file_signature(path):
    """Return (md5, phash, width, height, ok) for one image file."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        md5 = hashlib.md5(raw).hexdigest()
        with Image.open(path) as im:
            w, h = im.size
            ph = phash64_from_image(im)
        return md5, int(ph), w, h, True
    except Exception:
        return "", int(BAD_HASH), 0, 0, False


def popcount64(x):
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _POP8[x.view(np.uint8)].reshape(-1, 8).sum(axis=1)


def near_duplicate_pairs(hashes, threshold=5):
    """All index pairs (i < j) whose pHash Hamming distance is <= threshold.

    Multi-index hashing: the 64 bits are cut into threshold+1 blocks; by the
    pigeonhole principle two hashes within the threshold agree exactly on at least
    one block, so only hashes sharing a block value need to be compared.
    Returns {(i, j): distance}.
    """
    h = np.asarray(hashes, dtype=np.uint64)
    valid = h != BAD_HASH
    nblocks = threshold + 1
    widths = [64 // nblocks + (1 if k < 64 % nblocks else 0) for k in range(nblocks)]
    pairs = {}
    start = 0
    for width in widths:
        key = (h >> np.uint64(start)) & np.uint64((1 << width) - 1)
        start += width
        idx = np.flatnonzero(valid)
        order = idx[np.argsort(key[idx], kind="stable")]
        ks = key[order]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        for grp in np.split(order, bounds):
            if len(grp) < 2:
                continue
            hg = h[grp]
            for a in range(len(grp) - 1):
                d = popcount64(hg[a + 1:] ^ hg[a])
                for b in np.flatnonzero(d <= threshold):
                    i, j = int(grp[a]), int(grp[a + 1 + b])
                    if i > j:
                        i, j = j, i
                    pairs[(i, j)] = int(d[b])
    return pairs


def exact_duplicate_pairs(md5s):
    """Pairs of indices with identical file content -> {(i, j): 0}."""
    groups = defaultdict(list)
    for i, m in enumerate(md5s):
        if m:
            groups[m].append(i)
    pairs = {}
    for members in groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pairs[(members[a], members[b])] = 0
    return pairs


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
