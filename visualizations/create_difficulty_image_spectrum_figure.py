"""
Generate an ImageNet difficulty spectrum figure.

Inputs:
  - imagenet_examples_ammended.csv: Single comma-separated line of image paths in a global order
    of arbitrary length N. Placeholders like "None" may appear for filtered sets.
  - imagenet_synset_hierarchy.json (optional): Mapping from WNIDs to metadata, including "words"
    labels. If missing, WNIDs will be shown as labels.

Process (single-row spectrum):
  1) Consider a difficulty sub-range of global ranks [start_rank, end_rank).
  2) Split that range into NUM_BINS equal bins (default 12).
  3) Pick one image per bin near its center, preferring class diversity across the 12 picks.
  4) Render one row of 12 images; above it, render a colored difficulty scale spanning the range.

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
    """Return defaults for csv, hierarchy json, and output path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bars_dir = os.path.join(project_root, "bars")
    csv_path = os.path.join(bars_dir, "imagenet_examples_ammended.csv")
    hier_path = os.path.join(bars_dir, "imagenet_synset_hierarchy.json")
    out_path = os.path.join(os.path.dirname(__file__), "difficulty_spectrum.png")

    return csv_path, hier_path, out_path


def parse_args() -> argparse.Namespace:
    d_csv, d_hier, d_out = default_paths()
    p = argparse.ArgumentParser(description="Generate ImageNet difficulty spectrum figure.")
    p.add_argument("--csv", default=d_csv, help="Path to imagenet_examples_ammended.csv")
    p.add_argument("--hier", default=d_hier, help="Path to imagenet_synset_hierarchy.json")
    p.add_argument("--out", default=d_out, help="Output image path (PNG)")
    p.add_argument("--bins", type=int, default=12, help="Number of rank bins (images)")
    p.add_argument("--start_rank", type=int, default=1, help="Inclusive 1-based start rank (default 1)")
    p.add_argument("--end_rank", type=int, default=0, help="Exclusive 1-based end rank (0=end of file)")
    p.add_argument("--classes", type=int, default=5, help="Number of classes to prioritize for diversity")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed for tie-breaking")
    p.add_argument("--thumb", type=int, default=160, help="Thumbnail square size in pixels")
    p.add_argument("--dpi", type=int, default=350, help="Output figure DPI")
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


# No scores/NPY required for this visualization; selection relies solely on CSV ranks and class spread


def path_to_wnid(path: str) -> Optional[str]:
    # expects .../val/<wnid>/<file>.JPEG
    m = re.search(r"/(n\d{8})/", path.replace("\\", "/"))
    return m.group(1) if m else None


def continuous_colormap(width: int, cmap_name: str = "plasma") -> np.ndarray:
    """Return a 1xW RGB image representing a continuous colormap."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
    xs = np.linspace(0.0, 1.0, width)
    rgb = cmap(xs)[..., :3]
    return np.expand_dims(rgb, axis=0)


def load_hierarchy_labels(hier_path: str) -> Dict[str, str]:
    try:
        with open(hier_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {wnid: meta.get("words", wnid) for wnid, meta in data.items()}
    except Exception:
        return {}


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

    paths_all = parse_imagenet_examples_csv(args.csv)
    num_images = len(paths_all)
    if num_images <= 0:
        raise ValueError("CSV appears empty")

    labels = load_hierarchy_labels(args.hier)

    # Build wnid -> indices (0-based ranks)
    wnid_to_indices: Dict[str, List[int]] = {}
    for idx, p in enumerate(paths_all):
        if p == "None" or p == "":
            continue
        wnid = path_to_wnid(p)
        if wnid is None:
            continue
        wnid_to_indices.setdefault(wnid, []).append(idx)

    # Restrict to requested rank window [start_rank, end_rank) (1-based ranks)
    start_rank = max(1, int(args.start_rank))
    end_rank = int(args.end_rank) if int(args.end_rank) > 0 else num_images
    if end_rank <= start_rank:
        raise ValueError("end_rank must be greater than start_rank")
    start_idx = start_rank - 1
    end_idx_exclusive = end_rank  # already exclusive
    total_in_window = end_idx_exclusive - start_idx
    if total_in_window <= 0:
        raise ValueError("Empty rank window after conversion to 0-based indices")

    num_bins = int(args.bins)
    # Floating bin edges across the selected window
    bin_edges = [start_idx + i * (total_in_window / num_bins) for i in range(num_bins + 1)]
    bin_centers = [int((bin_edges[i] + bin_edges[i + 1]) / 2) for i in range(num_bins)]

    # Choose 5 classes with best uniform spread across this window
    def compute_class_score(indices: List[int]) -> float:
        counts = [0] * num_bins
        for idx in indices:
            if idx < start_idx or idx >= end_idx_exclusive:
                continue
            b = min(int((idx - start_idx) * num_bins / total_in_window), num_bins - 1)
            counts[b] += 1
        total = sum(counts)
        if total == 0:
            return float("inf")
        expected = total / num_bins
        chisq = sum(((c - expected) ** 2) / (expected + 1e-9) for c in counts)
        zeros = sum(1 for c in counts if c == 0)
        return chisq + zeros * expected

    class_scores: List[Tuple[str, float]] = []
    for wnid, idxs in wnid_to_indices.items():
        score = compute_class_score(idxs)
        class_scores.append((wnid, score))
    class_scores.sort(key=lambda t: t[1])
    chosen_wnids = [wnid for wnid, _ in class_scores[: max(1, int(args.classes))]]

    # Precompute per-bin nearest candidate for each chosen class
    per_class_bin_lists: Dict[str, List[List[int]]] = {w: [[] for _ in range(num_bins)] for w in chosen_wnids}
    for w in chosen_wnids:
        for idx in wnid_to_indices.get(w, []):
            if idx < start_idx or idx >= end_idx_exclusive:
                continue
            b = min(int((idx - start_idx) * num_bins / total_in_window), num_bins - 1)
            per_class_bin_lists[w][b].append(idx)
        for b in range(num_bins):
            per_class_bin_lists[w][b].sort()

    # Candidate indices per bin; choose one per bin with class-diversity preference among chosen classes
    # Build global index list for quick wnid lookup
    def idx_to_wnid(i: int) -> Optional[str]:
        p = paths_all[i]
        if not p or p == "None":
            return None
        return path_to_wnid(p)

    chosen_indices: List[int] = []
    used_wnids: Dict[str, int] = {w: 0 for w in chosen_wnids}
    for b in range(num_bins):
        center = bin_centers[b]
        # Build per-class nearest candidate within this bin
        candidates: List[Tuple[int, str, int]] = []  # (distance, wnid, idx)
        for w in chosen_wnids:
            lst = per_class_bin_lists[w][b]
            if not lst:
                continue
            # pick nearest to center
            nearest = min(lst, key=lambda j: abs(j - center))
            candidates.append((abs(nearest - center), w, nearest))
        if candidates:
            # Prefer classes used fewer times; then smaller distance
            candidates.sort(key=lambda t: (used_wnids.get(t[1], 0), t[0]))
            _, w_sel, idx_sel = candidates[0]
            chosen_indices.append(idx_sel)
            used_wnids[w_sel] = used_wnids.get(w_sel, 0) + 1
        else:
            # Fallback: search any class in window
            max_radius = int(math.ceil((bin_edges[b + 1] - bin_edges[b]) * 2))
            chosen_any: Optional[int] = None
            for radius in range(max_radius):
                for sign in (-1, 1):
                    j = center + sign * radius
                    if j < int(bin_edges[b]) or j >= int(bin_edges[b + 1]):
                        continue
                    if paths_all[j] and paths_all[j] != "None":
                        chosen_any = j
                        break
                if chosen_any is not None:
                    break
            if chosen_any is not None:
                chosen_indices.append(chosen_any)

    # Plot: one top difficulty bar + one row of 12 images
    cols = num_bins
    rows = 1
    fig_h = 3.8
    fig_w = 1.6 * cols
    fig, axes = plt.subplots(rows + 1, cols, figsize=(fig_w, fig_h), gridspec_kw={"height_ratios": [0.5] + [1] * rows})

    # Top difficulty colored scale
    gradient = continuous_colormap(800, cmap_name="plasma")
    for c in range(cols):
        axes[0, c].axis("off")
    # Draw gradient across a single invisible axis spanning all columns by overlaying on axes[0, 0]
    ax0 = axes[0, 0]
    ax0.imshow(gradient, aspect="auto", extent=[0, 1, 0, 1])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    # Add ticks at bin edges with rank labels
    for i in range(cols + 1):
        x = i / cols
        ax0.plot([x, x], [0.0, 1.0], color=(1, 1, 1, 0.5), linewidth=0.8)
        if i < cols:
            rank_right = int((bin_edges[i + 1]) + 1)  # convert to 1-based approx
            axes[0, i].text(0.5, -0.2, f"≤{rank_right}", ha="center", va="top", transform=axes[0, i].transAxes, fontsize=8)
    axes[0, 0].text(0.0, 1.25, f"Difficulty ({start_rank:,} → {end_rank:,})", transform=axes[0, 0].transAxes, fontsize=12, fontweight="bold")

    # One row of thumbnails
    thumb_size = (args.thumb, args.thumb)
    for col in range(cols):
        ax = axes[1, col]
        ax.axis("off")
        if col >= len(chosen_indices):
            continue
        idx = chosen_indices[col]
        img = load_image_or_tile(paths_all[idx], size=thumb_size)
        wnid = path_to_wnid(paths_all[idx]) or ""
        label = labels.get(wnid, wnid)
        ax.imshow(img)
        ax.set_title(f"{idx + 1}", fontsize=9)
        ax.text(0.5, -0.18, label, ha="center", va="top", fontsize=8, transform=ax.transAxes, wrap=True)

    plt.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi))
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()


