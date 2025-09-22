"""
Generate an ImageNet difficulty spectrum figure.

Inputs:
  - imagenet_examples_ammended.csv: Single comma-separated line of 50,000 validation image paths
    in the universal order (rank 0..49999). Placeholders like "None" may appear for filtered sets.
  - imagenet.npy: Model-by-image correctness matrix aligned to the same 50k order. This may be
    a boolean ndarray or an object ndarray containing booleans and Nones; we normalize to bool.
  - imagenet_synset_hierarchy.json: Mapping from WNIDs to metadata, including "words" labels.

Process:
  1) Build per-class index lists across the 50k ranks using WNIDs parsed from paths.
  2) Score classes by how uniformly their indices cover NUM_BINS bins across the 50k ranks.
  3) Choose TOP_K classes with lowest uniformity score (and at least 2*NUM_BINS examples).
  4) For each chosen class and each bin, sample two images closest to the bin center.
  5) Render a grid: a top difficulty axis row; then one row per class with 2 images per bin.

Usage:
  python visualizations/main.py \
    --csv bars/imagenet_examples_ammended.csv \
    --npy bars/imagenet.npy \
    --hier bars/imagenet_synset_hierarchy.json \
    --out visualizations/difficulty_spectrum.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def default_paths() -> Tuple[str, str, str, str]:
    """Return defaults for csv, npy, hierarchy json, and output path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bars_dir = os.path.join(project_root, "bars")
    csv_path = os.path.join(bars_dir, "imagenet_examples_ammended.csv")
    npy_path = os.path.join(bars_dir, "imagenet.npy")
    hier_path = os.path.join(bars_dir, "imagenet_synset_hierarchy.json")
    out_path = os.path.join(os.path.dirname(__file__), "difficulty_spectrum.png")

    # Prefer filtered versions if present
    try:
        for candidate in sorted(os.listdir(bars_dir)):
            if candidate.endswith("imagenet_examples_ammended.csv") and candidate != "imagenet_examples_ammended.csv":
                csv_path = os.path.join(bars_dir, candidate)
            if candidate.endswith("imagenet.npy") and candidate != "imagenet.npy":
                npy_path = os.path.join(bars_dir, candidate)
    except FileNotFoundError:
        pass
    return csv_path, npy_path, hier_path, out_path


# Module-level defaults computed once
DEFAULT_CSV, DEFAULT_NPY, DEFAULT_HIER, DEFAULT_OUT = default_paths()
DEFAULT_BINS = 12
DEFAULT_CLASSES = 5
DEFAULT_SEED = 1337
DEFAULT_THUMB = 128


def parse_args() -> argparse.Namespace:
    d_csv, d_npy, d_hier, d_out = DEFAULT_CSV, DEFAULT_NPY, DEFAULT_HIER, DEFAULT_OUT
    p = argparse.ArgumentParser(description="Generate ImageNet difficulty spectrum figure.")
    p.add_argument("--csv", default=d_csv, help="Path to imagenet_examples_ammended.csv")
    p.add_argument("--npy", default=d_npy, help="Path to imagenet.npy")
    p.add_argument("--hier", default=d_hier, help="Path to imagenet_synset_hierarchy.json")
    p.add_argument("--out", default=d_out, help="Output image path (PNG)")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of rank bins")
    p.add_argument("--classes", type=int, default=DEFAULT_CLASSES, help="Number of classes to show")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for tie-breaking")
    p.add_argument("--thumb", type=int, default=DEFAULT_THUMB, help="Thumbnail square size in pixels")
    return p.parse_args()


def parse_imagenet_examples_csv(csv_path: str) -> List[str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        text = f.read().lstrip("\ufeff").strip()
    if not text:
        return []
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p != ""]


def load_correct_matrix(npy_path: str, expected_images: int | None = None) -> np.ndarray:
    arr = np.load(npy_path, allow_pickle=True)
    if arr.dtype == object:
        if arr.ndim == 1:
            rows: List[np.ndarray] = []
            for row in arr.tolist():
                rows.append(np.asarray([False if (x is None) else bool(x) for x in row], dtype=bool))
            matrix = np.vstack(rows)
        else:
            m, n = arr.shape
            matrix = np.zeros((m, n), dtype=bool)
            for i in range(m):
                for j in range(n):
                    x = arr[i, j]
                    matrix[i, j] = False if (x is None) else bool(x)
    else:
        matrix = arr.astype(bool) if arr.dtype != bool else arr
    if matrix.ndim == 1:
        matrix = np.expand_dims(matrix, axis=0)
    if expected_images is not None and matrix.shape[1] != expected_images:
        raise ValueError(f"NPY images={matrix.shape[1]} mismatch vs expected {expected_images}")
    return matrix


def path_to_wnid(path: str) -> Optional[str]:
    # expects .../val/<wnid>/<file>.JPEG
    m = re.search(r"/(n\d{8})/", path.replace("\\", "/"))
    return m.group(1) if m else None


@dataclass
class ClassCoverage:
    wnid: str
    label: str
    counts: List[int]
    score: float


def compute_coverage(indices: List[int], num_images: int, num_bins: int) -> Tuple[List[int], float]:
    counts = [0] * num_bins
    for idx in indices:
        b = min(int(idx * num_bins / num_images), num_bins - 1)
        counts[b] += 1
    total = sum(counts)
    if total == 0:
        return counts, float("inf")
    expected = total / num_bins
    chisq = sum(((c - expected) ** 2) / (expected + 1e-9) for c in counts)
    zeros = sum(1 for c in counts if c == 0)
    score = chisq + zeros * expected
    return counts, float(score)


def load_hierarchy_labels(hier_path: str) -> Dict[str, str]:
    with open(hier_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {wnid: meta.get("words", wnid) for wnid, meta in data.items()}


def load_image_or_tile(path: str, size: Tuple[int, int]) -> Image.Image:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail(size, Image.Resampling.LANCZOS)
            bg = Image.new("RGB", size, (255, 255, 255))
            bg.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
            return bg
    except Exception:
        return Image.new("RGB", size, (220, 220, 220))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    paths_50k = parse_imagenet_examples_csv(args.csv)
    num_images = len(paths_50k)
    if num_images != 50000:
        raise ValueError(f"Expected 50,000 images in CSV, got {num_images}")

    correct_matrix = load_correct_matrix(args.npy, expected_images=num_images)
    _num_models, _ = correct_matrix.shape

    labels = load_hierarchy_labels(args.hier)

    # Build wnid -> indices
    wnid_to_indices: Dict[str, List[int]] = {}
    for idx, p in enumerate(paths_50k):
        if p == "None" or p == "":
            continue
        wnid = path_to_wnid(p)
        if wnid is None:
            continue
        wnid_to_indices.setdefault(wnid, []).append(idx)

    num_bins = args.bins
    bin_edges = [i * num_images / num_bins for i in range(num_bins + 1)]
    bin_centers = [int((bin_edges[i] + bin_edges[i + 1]) / 2) for i in range(num_bins)]

    coverages: List[ClassCoverage] = []
    for wnid, idxs in wnid_to_indices.items():
        counts, score = compute_coverage(idxs, num_images=num_images, num_bins=num_bins)
        coverages.append(ClassCoverage(wnid=wnid, label=labels.get(wnid, wnid), counts=counts, score=score))

    min_per_class = 2 * num_bins
    candidates = [c for c in coverages if sum(c.counts) >= min_per_class]
    random.shuffle(candidates)
    candidates.sort(key=lambda c: c.score)
    chosen = candidates[: args.classes]

    # Prepare per-bin index lists for chosen classes
    class_bin_indices: Dict[str, List[List[int]]] = {}
    for c in chosen:
        per_bin = [[] for _ in range(num_bins)]
        for idx in wnid_to_indices[c.wnid]:
            b = min(int(idx * num_bins / num_images), num_bins - 1)
            per_bin[b].append(idx)
        for b in range(num_bins):
            per_bin[b].sort()
        class_bin_indices[c.wnid] = per_bin

    # Plot
    cols = num_bins * 2
    rows = len(chosen)
    fig_h = 2.2 * rows
    fig_w = 1.8 * cols / 6
    fig, axes = plt.subplots(rows + 1, cols, figsize=(fig_w, fig_h), gridspec_kw={"height_ratios": [0.6] + [1] * rows})

    # Top difficulty axis
    for col in range(cols):
        ax = axes[0, col]
        ax.axis("off")
        bin_id = col // 2
        if col % 2 == 0:
            ax.text(0.5, 0.5, f"≤{int(bin_edges[bin_id + 1])}", ha="center", va="center", fontsize=9)
    axes[0, 0].text(0.0, 1.2, "Difficulty (easy → hard)", transform=axes[0, 0].transAxes, fontsize=12, fontweight="bold")

    # Rows per class with thumbnails
    thumb_size = (args.thumb, args.thumb)
    for r, c in enumerate(chosen, start=1):
        label_ax = axes[r, 0]
        label_ax.text(-0.6, 0.5, c.label, va="center", ha="right", fontsize=11, transform=label_ax.transAxes)
        for col in range(cols):
            axes[r, col].axis("off")

        per_bin_used = {b: 0 for b in range(num_bins)}
        per_bin = class_bin_indices[c.wnid]
        for b in range(num_bins):
            idxs = per_bin[b]
            if not idxs:
                continue
            center = bin_centers[b]
            for idx in sorted(idxs, key=lambda i: abs(i - center))[:2]:
                slot = per_bin_used[b]
                if slot >= 2:
                    break
                col = b * 2 + slot
                img = load_image_or_tile(paths_50k[idx], size=thumb_size)
                axes[r, col].imshow(img)
                axes[r, col].set_title(f"{idx + 1}", fontsize=8)
                per_bin_used[b] += 1

    plt.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()


