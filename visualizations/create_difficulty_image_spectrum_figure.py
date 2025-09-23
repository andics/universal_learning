"""
Generate an ImageNet difficulty spectrum figure.

Inputs:
  - imagenet_examples_ammended.csv: Single comma-separated line of image paths in a global order
    of arbitrary length N. Placeholders like "None" may appear for filtered sets.
  - imagenet_synset_hierarchy.json (optional): Mapping from WNIDs to metadata, including "words"
    labels. If missing, WNIDs will be shown as labels.
  - Fixed WNIDs can be provided (defaults to: Piano n03452741, Zebra n02391049, Water bottle n04557648,
    Wooden spoon n04597913, Dial phone n03187595).

Process (multi-row spectrum):
  1) Determine the difficulty range from the first to the last non-None path in the CSV
     (1-based ranks). Optionally override with --start_rank/--end_rank.
  2) Compute NUM_BINS (=12) equal bins across that range.
  3) Select K (=5) classes whose examples are most uniformly distributed across bins
     in this range (must have at least 12 examples inside the range).
  4) For each of the K classes, pick one image per bin (nearest to bin center; fallback to
     nearest-in-range when a bin is empty for that class), producing 12 images per class.
  5) Render a figure with a colored difficulty spectrum at the top, and K rows below:
     leftmost column is the class label, followed by 12 image columns (13 columns total).

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
    p.add_argument("--bins", type=int, default=10, help="Number of rank bins (images/pairs*2)")
    p.add_argument("--start_rank", type=int, default=1, help="Inclusive 1-based start rank (default 1)")
    p.add_argument("--end_rank", type=int, default=0, help="Exclusive 1-based end rank (0=end of file)")
    p.add_argument("--classes", type=int, default=5, help="Number of classes (rows) to show")
    p.add_argument(
        "--wnids",
        type=str,
        default="n03452741,n02391049,n04557648,n04597913,n03187595",
        help="Comma-separated list of WNIDs to visualize (default: piano,zebra,water bottle,wooden spoon,dial phone)",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for tie-breaking (None=random)")
    p.add_argument("--thumb", type=int, default=160, help="Thumbnail square size in pixels")
    p.add_argument("--dpi", type=int, default=350, help="Output figure DPI")
    p.add_argument("--pair_gap", type=int, default=12, help="Horizontal gap (pixels) between image pairs")
    p.add_argument(
        "--second_wnids",
        type=str,
        default="n03691459,n03452741,n03187595,n03481172,n03637318",
        help="WNIDs for an additional collage (speaker,piano,dial phone,hammer,lampshade)",
    )
    p.add_argument(
        "--second_out",
        type=str,
        default=None,
        help="Output path for the second collage (default: derive from --out by appending _second)",
    )
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
            # Letterbox to exact size with black margins
            w, h = img.size
            target_w, target_h = size
            scale = min(target_w / w, target_h / h) if (w > 0 and h > 0) else 1.0
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            bg = Image.new("RGB", size, (0, 0, 0))
            bg.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
            return bg
    except Exception:
        return Image.new("RGB", size, (0, 0, 0))


def build_and_save_collage(paths_all: List[str], labels: Dict[str, str], args: argparse.Namespace, requested_wnids_csv: str, out_path: str) -> None:
    # New seed per run if None, to change picks while staying near bin centers
    if args.seed is None:
        random.seed()
    else:
        random.seed(args.seed)

    paths_all = parse_imagenet_examples_csv(args.csv)
    num_images = len(paths_all)
    if num_images <= 0:
        raise ValueError("CSV appears empty")

    # labels mapping already provided

    # Build wnid -> indices (0-based ranks)
    wnid_to_indices: Dict[str, List[int]] = {}
    for idx, p in enumerate(paths_all):
        if p == "None" or p == "":
            continue
        wnid = path_to_wnid(p)
        if wnid is None:
            continue
        wnid_to_indices.setdefault(wnid, []).append(idx)

    # Compute difficulty window from first and last non-None path in the CSV
    first_idx = None
    last_idx = None
    for i, p in enumerate(paths_all):
        if p and p != "None":
            first_idx = i
            break
    for i in range(len(paths_all) - 1, -1, -1):
        p = paths_all[i]
        if p and p != "None":
            last_idx = i
            break
    if first_idx is None or last_idx is None or last_idx < first_idx:
        raise ValueError("Could not determine non-None range from CSV")
    start_idx = first_idx
    end_idx_exclusive = last_idx + 1  # exclusive
    start_rank = start_idx + 1  # for display (1-based)
    end_rank = last_idx + 1
    total_in_window = end_idx_exclusive - start_idx
    if total_in_window <= 0:
        raise ValueError("Empty rank window after conversion to 0-based indices")

    num_bins = int(args.bins)
    # Floating bin edges across the selected window
    bin_edges = [start_idx + i * (total_in_window / num_bins) for i in range(num_bins + 1)]
    bin_centers = [int((bin_edges[i] + bin_edges[i + 1]) / 2) for i in range(num_bins)]

    # Choose K classes with best uniform spread across this window
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

    # Use explicit WNIDs provided via args (defaults to the requested five)
    requested_wnids = [w.strip() for w in str(requested_wnids_csv).split(",") if w.strip()]
    # Filter to those present in CSV and with at least 1 sample in window
    present_wnids: List[str] = []
    for w in requested_wnids:
        idxs = wnid_to_indices.get(w, [])
        in_window = [i for i in idxs if start_idx <= i < end_idx_exclusive]
        if len(in_window) > 0:
            present_wnids.append(w)
    # If fewer than requested are present, fall back to best-uniform classes to fill the remainder
    chosen_wnids: List[str] = list(present_wnids)
    if len(chosen_wnids) < int(args.classes):
        class_scores: List[Tuple[str, float]] = []
        for wnid, idxs in wnid_to_indices.items():
            if wnid in chosen_wnids:
                continue
            in_window = [i for i in idxs if start_idx <= i < end_idx_exclusive]
            if len(in_window) < num_bins:
                continue
            score = compute_class_score(in_window)
            class_scores.append((wnid, score))
        class_scores.sort(key=lambda t: t[1])
        for wnid, _ in class_scores:
            if len(chosen_wnids) >= int(args.classes):
                break
            chosen_wnids.append(wnid)

    # Helper to choose nearest unused index to a target from a sorted list
    def nearest_unused(sorted_idxs: List[int], target: int, used: set[int]) -> Optional[int]:
        import bisect
        n = len(sorted_idxs)
        if n == 0:
            return None
        pos = bisect.bisect_left(sorted_idxs, target)
        left = pos - 1
        right = pos
        best_idx = None
        best_dist = None
        while left >= 0 or right < n:
            cand = None
            if left >= 0 and (right >= n or abs(sorted_idxs[left] - target) <= abs(sorted_idxs[right] - target)):
                cand = sorted_idxs[left]
                left -= 1
            elif right < n:
                cand = sorted_idxs[right]
                right += 1
            if cand is None:
                break
            if cand not in used:
                d = abs(cand - target)
                if best_dist is None or d < best_dist:
                    best_idx = cand
                    best_dist = d
                    break
        if best_idx is None:
            # fallback any unused
            for cand in sorted_idxs:
                if cand not in used:
                    return cand
        return best_idx

    # For each chosen class, collect 12 uniformly spaced images (nearest to bin center)
    class_to_samples: Dict[str, List[Optional[int]]] = {}
    for w in chosen_wnids:
        indices_sorted = sorted([i for i in wnid_to_indices.get(w, []) if start_idx <= i < end_idx_exclusive])
        # Build per-bin candidate lists sorted by distance to center
        bin_candidates: List[List[int]] = [[] for _ in range(num_bins)]
        for idx in indices_sorted:
            b = min(int((idx - start_idx) * num_bins / total_in_window), num_bins - 1)
            bin_candidates[b].append(idx)
        for b in range(num_bins):
            bin_candidates[b].sort(key=lambda j: abs(j - bin_centers[b]))
        # Greedy assignment with stochastic tie-breaking to vary picks across runs
        rng = random.Random()
        order = list(range(num_bins))
        order.sort(key=lambda b: (len(bin_candidates[b]), b))
        used: set[int] = set()
        picks_by_bin: List[Optional[int]] = [None] * num_bins
        for b in order:
            cands = bin_candidates[b]
            if not cands:
                continue
            # among top-k closest (k up to 3), pick randomly for variety
            k = min(3, len(cands))
            top_k = cands[:k]
            top_k = [j for j in top_k if j not in used]
            if top_k:
                j = rng.choice(top_k)
                picks_by_bin[b] = j
                used.add(j)
                continue
            # otherwise pick the nearest unused
            for j in cands:
                if j not in used:
                    picks_by_bin[b] = j
                    used.add(j)
                    break
        # Fallback for bins not filled: take nearest from remaining indices
        remain = [i for i in indices_sorted if i not in used]
        for b in range(num_bins):
            if picks_by_bin[b] is not None:
                continue
            if not remain:
                break
            # pick nearest remaining to center
            j = min(remain, key=lambda x: abs(x - bin_centers[b]))
            picks_by_bin[b] = j
            remain.remove(j)
        class_to_samples[w] = picks_by_bin

    # Plot: top difficulty spectrum + K class rows, label + images + small gaps between pairs
    rows = len(chosen_wnids)
    tile = int(args.thumb)
    gap_px = max(0, int(args.pair_gap))
    top_bar_height = int(tile * 0.6)

    # Build width ratios: [label] + for b in bins: [img] and insert [gap] after each pair except last
    def bin_to_col_index(b: int) -> int:
        # label is column 0; number of gaps before bin b is floor(b/2)
        return 1 + b + (b // 2)

    width_ratios: List[float] = []
    width_ratios.append(tile)  # label column
    for b in range(num_bins):
        width_ratios.append(tile)
        # after each pair except the last pair, add a narrow gap column
        if b % 2 == 1 and b < num_bins - 1:
            width_ratios.append(gap_px)

    cols_total = len(width_ratios)
    fig_w = (sum(width_ratios)) / 100.0
    fig_h = ((rows * tile) + top_bar_height) / 100.0
    fig, axes = plt.subplots(
        rows + 1,
        cols_total,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [top_bar_height] + [tile] * rows, "width_ratios": width_ratios},
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
    try:
        fig.patch.set_facecolor("black")
    except Exception:
        pass

    # Top difficulty colored scale spanning over image columns (1..num_bins)
    for c in range(cols_total):
        axes[0, c].axis("off")
    # compute combined bbox for axes[0, first_img]..axes[0, last_img]
    first_img_col = bin_to_col_index(0)
    last_img_col = bin_to_col_index(num_bins - 1)
    left = axes[0, first_img_col].get_position().x0
    right = axes[0, last_img_col].get_position().x1
    bottom = axes[0, 1].get_position().y0
    top = axes[0, 1].get_position().y1
    overlay_ax = fig.add_axes([left, bottom, right - left, top - bottom])
    gradient = continuous_colormap(1200, cmap_name="plasma")
    overlay_ax.imshow(gradient, aspect="auto", extent=[0, 1, 0, 1])
    overlay_ax.set_xlim(0, 1)
    overlay_ax.set_ylim(0, 1)
    overlay_ax.axis("off")
    for i in range(num_bins + 1):
        x = i / num_bins
        overlay_ax.plot([x, x], [0.0, 1.0], color=(1, 1, 1, 0.5), linewidth=0.8)
    # Title and numeric labels every two bars (i=0,2,4,6,8,10)
    axes[0, first_img_col].text(0.0, 1.25, f"Difficulty ({start_rank:,} → {end_rank:,})", transform=axes[0, first_img_col].transAxes, fontsize=12, fontweight="bold")
    bar_size = (end_rank - start_rank) / float(num_bins)
    for j in range(0, num_bins + 1, 2):
        x = j / num_bins
        label_val = int(round(start_rank + j * bar_size))
        overlay_ax.text(x, -0.15, f"{label_val:,}", ha="center", va="top", fontsize=8, color="white", transform=overlay_ax.transAxes, clip_on=False)

    # Rows of thumbnails: each class per row, label column then 12 images
    thumb_size = (args.thumb, args.thumb)
    for r, w in enumerate(chosen_wnids, start=1):
        # label at column 0
        label_text = labels.get(w, w)
        axes[r, 0].axis("off")
        axes[r, 0].text(1.0, 0.5, label_text, ha="right", va="center", fontsize=11, transform=axes[r, 0].transAxes, wrap=True)
        # images
        picks = class_to_samples.get(w, [])
        for b in range(num_bins):
            col_idx = bin_to_col_index(b)
            ax = axes[r, col_idx]
            ax.axis("off")
            if b >= len(picks) or picks[b] is None:
                continue
            idx = int(picks[b])
            img = load_image_or_tile(paths_all[idx], size=thumb_size)
            ax.set_facecolor("black")
            ax.imshow(img, aspect="auto")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    print(f"Saved figure to: {out_path}")


def main() -> None:
    args = parse_args()
    # New seed per run if None, to change picks while staying near bin centers
    if args.seed is None:
        random.seed()
    else:
        random.seed(args.seed)

    paths_all = parse_imagenet_examples_csv(args.csv)
    num_images = len(paths_all)
    if num_images <= 0:
        raise ValueError("CSV appears empty")

    labels = load_hierarchy_labels(args.hier)

    # First collage
    build_and_save_collage(paths_all, labels, args, args.wnids, args.out)

    # Second collage if requested
    if args.second_out is None:
        root, ext = os.path.splitext(args.out)
        second_out = f"{root}_second{ext or '.png'}"
    else:
        second_out = args.second_out
    if args.second_wnids:
        build_and_save_collage(paths_all, labels, args, args.second_wnids, second_out)


if __name__ == "__main__":
    main()


