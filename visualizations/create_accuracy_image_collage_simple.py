from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bars_dir = os.path.join(project_root, "bars")
    d_csv = os.path.join(bars_dir, "imagenet_examples_ammended.csv")
    d_npy = os.path.join(bars_dir, "imagenet.npy")
    d_out = os.path.join(os.path.dirname(__file__), "difficulty_spectrum_third_5.png")
    d_chosen = os.path.join(os.path.dirname(__file__), "collage_3_chosen_granularity")

    p = argparse.ArgumentParser(description="Build a simple collage of selected images with per-image accuracy from imagenet.npy.")
    p.add_argument("--csv", default=d_csv, help="Path to imagenet_examples_ammended.csv (single row CSV of 50k paths)")
    p.add_argument("--npy", default=d_npy, help="Path to imagenet.npy (shape [num_models, num_images])")
    p.add_argument("--out", default=d_out, help="Output collage image path")
    p.add_argument("--root_dir", type=str, default=None, help="Optional root dir to prefix to CSV paths to load images")
    p.add_argument("--thumb", type=int, default=160, help="Thumbnail size (square side in px)")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI")
    p.add_argument("--pair_gap", type=int, default=12, help="Gap between image pairs (px)")
    p.add_argument("--font_size", type=int, default=6, help="Tiny font size for accuracy text (used when possible)")
    p.add_argument("--threshold", type=float, default=None, help="Optional threshold to convert float scores to correctness; if None, infer from dtype")
    p.add_argument("--wnids", type=str, default="n03187595,n03452741,n03481172,n03637318,n02504458", help="Comma-separated wnids; images will be drawn in listed order of each wnid's mapping list")
    p.add_argument("--chosen_dir", type=str, default=d_chosen, help="Folder containing 5 subfolders (rows) with images named with leading rank (default: collage_3_chosen_images next to this script)")
    return p.parse_args()


def read_single_row_csv_paths(csv_path: str) -> List[str]:
    with open(csv_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if "\n" not in text and "," in text:
        parts = [p.strip() for p in text.split(",")]
    else:
        # Fallback: use CSV reader for robustness
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        parts = []
        for row in rows:
            parts.extend([p.strip() for p in row])
    return [p for p in parts if p != ""]


def load_imagenet_correctness(npy_path: str, threshold: Optional[float]) -> np.ndarray:
    arr = np.load(npy_path, allow_pickle=True)

    if arr.dtype == object:
        if arr.ndim == 1:
            rows = []
            for row in arr.tolist():
                bool_row = [False if (x is None) else bool(x) for x in row]
                rows.append(np.asarray(bool_row, dtype=bool))
            M = np.vstack(rows)
        elif arr.ndim == 2:
            num_models, num_images = arr.shape
            M = np.zeros((num_models, num_images), dtype=bool)
            for i in range(num_models):
                for j in range(num_images):
                    x = arr[i, j]
                    M[i, j] = False if (x is None) else bool(x)
        else:
            raise ValueError(f"Unexpected object array shape: {arr.shape}")
        return M.astype(np.uint8)

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0)
        thr = 0.5 if threshold is None else float(threshold)
        M = (arr > thr).astype(np.uint8)
        return M

    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)

    # Other integer/binary encodings: treat nonzero as True
    return (arr != 0).astype(np.uint8)


def compute_accuracy_for_indices(correct_matrix: np.ndarray, indices: List[int]) -> Dict[int, Tuple[int, int, float]]:
    # Returns mapping: index -> (num_true, num_models, acc)
    result: Dict[int, Tuple[int, int, float]] = {}
    if correct_matrix.ndim != 2:
        raise ValueError(f"Expected 2D correctness matrix, got {correct_matrix.shape}")
    num_models = correct_matrix.shape[0]
    for j in indices:
        col = correct_matrix[:, j]
        num_true = int(col.sum())
        acc = float(num_true) / float(num_models) if num_models > 0 else 0.0
        result[j] = (num_true, num_models, acc)
    return result


def open_image_rgba_centered(path: str, size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            # Resize preserving aspect ratio to fit within size
            im.thumbnail((target_w, target_h), Image.LANCZOS)
            new_w, new_h = im.size
            bg = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 255))
            bg.paste(im, ((target_w - new_w) // 2, (target_h - new_h) // 2))
            return bg
    except Exception:
        return Image.new("RGBA", (target_w, target_h), (0, 0, 0, 255))


def build_pairs_collage_with_bg(
    per_class_indices: Dict[str, List[int]],
    per_index_accuracy: Dict[int, Tuple[int, int, float]],
    all_paths: List[str],
    out_path: str,
    thumb: int,
    pair_gap: int,
    dpi: int,
    tiny_font_size: int,
    root_dir: Optional[str],
) -> None:
    # Match the layout and style of collage 3: rows of 5 pairs (10 images) per class,
    # gradient background (custom_yellow_blue_fixed), transparent figure, black border around pairs.
    from PIL import ImageDraw, ImageFont

    chosen_wnids = list(per_class_indices.keys())
    rows = len(chosen_wnids)
    tile = int(thumb)
    gap_px = max(0, int(pair_gap))
    top_bar_height = int(tile * 0.6)

    num_bins = 10
    num_pairs = max(1, num_bins // 2)
    gap_w = int(round(gap_px * 1.5))
    gap_h = int(round(gap_px * 1.5))
    pair_width = tile * 2

    def pair_to_col_index(pair_id: int) -> int:
        return pair_id * 2

    width_ratios: List[float] = []
    for p in range(num_pairs):
        width_ratios.append(pair_width)
        if p < num_pairs - 1:
            width_ratios.append(gap_w)
    cols_total = len(width_ratios)

    height_ratios: List[float] = [top_bar_height]
    for r in range(rows):
        height_ratios.append(tile)
        if r < rows - 1:
            height_ratios.append(gap_h)

    fig_w = (sum(width_ratios)) / 100.0
    fig_h = (sum(height_ratios)) / 100.0
    fig, axes = plt.subplots(
        len(height_ratios),
        cols_total,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": height_ratios, "width_ratios": width_ratios},
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
    try:
        fig.patch.set_alpha(0)
        fig.patch.set_facecolor("none")
    except Exception:
        pass

    # turn off axes
    for r in range(len(height_ratios)):
        for c in range(cols_total):
            try:
                axes[r, c].set_axis_off()
            except Exception:
                pass

    # Background gradient matching custom_yellow_blue_fixed, but with white in the middle
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-100)
    def three_color_gradient_rgb(width: int, left_rgb: Tuple[int, int, int], mid_rgb: Tuple[int, int, int], right_rgb: Tuple[int, int, int]) -> np.ndarray:
        width = max(4, int(width))
        half = width // 2
        # left -> middle
        x1 = np.linspace(0.0, 1.0, half, dtype=np.float32)
        lr = np.array(left_rgb, dtype=np.float32)
        mr = np.array(mid_rgb, dtype=np.float32)
        grad1 = lr[None, :] * (1.0 - x1[:, None]) + mr[None, :] * x1[:, None]
        # middle -> right
        x2 = np.linspace(0.0, 1.0, width - half, dtype=np.float32)
        rr = np.array(right_rgb, dtype=np.float32)
        grad2 = mr[None, :] * (1.0 - x2[:, None]) + rr[None, :] * x2[:, None]
        grad = np.concatenate([grad1, grad2], axis=0)
        grad = np.clip(grad / 255.0, 0.0, 1.0)
        return np.tile(grad[None, :, :], (2, 1, 1))

    # left yellow (#FFDD59), middle white, right blue (#80A9FF)
    gradient = three_color_gradient_rgb(2000, (255, 221, 89), (255, 255, 255), (128, 169, 255))
    bg_ax.imshow(gradient, aspect="auto", extent=[0, 1, 0, 1], alpha=0.75)
    bg_ax.set_axis_off()

    # Ensure content axes above background
    try:
        for r in range(len(height_ratios)):
            for c in range(cols_total):
                axes[r, c].set_zorder(10)
    except Exception:
        pass

    # Font for tiny text: try to use a scalable font at ~20% larger than requested
    def load_font_scaled(base_px: int) -> Optional[ImageFont.ImageFont]:
        size = max(1, int(round(base_px * 2.0)))
        try:
            # Try DejaVu Sans from matplotlib
            import matplotlib.font_manager as fm
            font_path = fm.findfont('DejaVu Sans', fallback_to_default=True)
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            try:
                return ImageFont.load_default()
            except Exception:
                return None
    font = load_font_scaled(tiny_font_size)

    single_size = (tile, tile)
    pair_size = (pair_width, tile)

    def to_full_path(p: str) -> str:
        return os.path.join(root_dir, p.lstrip("/\\")) if root_dir else p

    def compose_pair(left_idx: Optional[int], right_idx: Optional[int]) -> Image.Image:
        left_img = Image.new("RGBA", single_size, (0, 0, 0, 255)) if left_idx is None else open_image_rgba_centered(to_full_path(all_paths[left_idx]), single_size)
        right_img = Image.new("RGBA", single_size, (0, 0, 0, 255)) if right_idx is None else open_image_rgba_centered(to_full_path(all_paths[right_idx]), single_size)
        pair_img = Image.new("RGBA", pair_size, (0, 0, 0, 0))
        pair_img.paste(left_img, (0, 0))
        pair_img.paste(right_img, (tile, 0))

        # Prepare drawing context and draw border first
        draw = ImageDraw.Draw(pair_img)
        border_px = max(1, int(round(tile * 0.02)))
        for k in range(border_px):
            draw.rectangle([k, k, pair_size[0]-1-k, pair_size[1]-1-k], outline=(0,0,0,255))

        # draw tiny accuracy text bottom-centered on each half, above the border
        def text_for(index: Optional[int]) -> str:
            if index is None:
                return ""
            nt, nm, acc = per_index_accuracy[int(index)]
            return f"{nt}/{nm} ({acc*100:.1f}%)"

        left_text = text_for(left_idx)
        right_text = text_for(right_idx)
        # bottom-centered positions
        padding_px = max(2, int(round(tile * 0.05)))
        def draw_outlined_text_bottom_center(x_center: int, text: str) -> None:
            if not text:
                return
            try:
                w, h = draw.textsize(text, font=font) if font else draw.textsize(text)
            except Exception:
                w, h = (len(text) * 3, 6)
            x = int(round(x_center - (w / 2.0)))
            # keep fully inside and above the inner border
            y = int(tile - border_px - padding_px - h)
            y = max(border_px + 1, y)
            # outline
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                draw.text((x+dx, y+dy), text, fill=(0,0,0,255), font=font)
            draw.text((x, y), text, fill=(255,255,255,255), font=font)

        # Bottom-center for each half
        draw_outlined_text_bottom_center(tile // 2, left_text)
        draw_outlined_text_bottom_center(tile + (tile // 2), right_text)

        return pair_img

    # place pairs
    for r_idx, w in enumerate(chosen_wnids):
        axis_row = 1 + r_idx * 2 if rows > 1 else 1
        picks = per_class_indices.get(w, [])
        for p in range(num_pairs):
            left_bin = p * 2
            right_bin = left_bin + 1
            left_idx = int(picks[left_bin]) if left_bin < len(picks) and picks[left_bin] is not None else None
            right_idx = int(picks[right_bin]) if right_bin < len(picks) and picks[right_bin] is not None else None
            col_idx = pair_to_col_index(p)
            ax = axes[axis_row, col_idx]
            ax.set_axis_off()
            pair_img = compose_pair(left_idx, right_idx)
            ax.set_facecolor("none")
            ax.imshow(pair_img, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            try:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            except Exception:
                pass

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), transparent=True)
    # Save PDF copy alongside PNG
    try:
        root, _ext = os.path.splitext(out_path)
        pdf_path = f"{root}.pdf"
        fig.savefig(pdf_path, dpi=int(dpi), transparent=True)
    except Exception:
        pass
    plt.close(fig)


def main() -> None:
    args = parse_args()

    # Read universal order paths
    paths_all = read_single_row_csv_paths(args.csv)
    if len(paths_all) <= 0:
        raise ValueError("CSV appears empty")

    # Load correctness matrix (models x images)
    C = load_imagenet_correctness(args.npy, args.threshold)

    if C.shape[1] != len(paths_all):
        raise ValueError(f"imagenet.npy columns ({C.shape[1]}) must equal CSV paths ({len(paths_all)})")

    # Resolve indices based on chosen_dir if provided, otherwise use explicit fixed list as before
    resolved_indices: List[int] = []
    resolved_paths: List[str] = []

    if args.chosen_dir and os.path.isdir(args.chosen_dir):
        # Expect 5 subfolders, each containing images named with leading rank like "1234_filename.JPEG"
        subdirs = [os.path.join(args.chosen_dir, d) for d in os.listdir(args.chosen_dir) if os.path.isdir(os.path.join(args.chosen_dir, d))]
        subdirs.sort()
        # For each subfolder, read files, parse ranks, keep order by rank ascending
        per_class_indices: List[List[int]] = []
        for sd in subdirs:
            files = [f for f in os.listdir(sd) if not f.startswith('.')]
            ranks: List[int] = []
            for fname in files:
                # Parse leading integer rank until first non-digit
                num = []
                for ch in fname:
                    if ch.isdigit():
                        num.append(ch)
                    else:
                        break
                if not num:
                    continue
                rank1 = int("".join(num))
                idx0 = rank1 - 1
                if 0 <= idx0 < len(paths_all):
                    ranks.append(idx0)
            ranks = sorted(set(ranks))
            # keep only first 10 (collage expects 10 per row)
            per_class_indices.append(ranks[:10])

        # Flatten in class order (preserve subdir order) to resolved_* lists
        for ranks in per_class_indices:
            for j in ranks:
                resolved_indices.append(j)
                resolved_paths.append(paths_all[j])
    else:
        # Fallback: original explicit list
        explicit = [
            (1203, "ILSVRC2012_val_00034672"),(1903, "ILSVRC2012_val_00022385"),(10502, "ILSVRC2012_val_00029045"),(13362, "ILSVRC2012_val_00013729"),(22345, "ILSVRC2012_val_00016388"),(22559, "ILSVRC2012_val_00029370"),(31523, "ILSVRC2012_val_00017123"),(36050, "ILSVRC2012_val_00033592"),(42722, "ILSVRC2012_val_00016249"),(43886, "ILSVRC2012_val_00000137"),
            (2581, "ILSVRC2012_val_00021661"),(6089, "ILSVRC2012_val_00007940"),(13804, "ILSVRC2012_val_00013511"),(13855, "ILSVRC2012_val_00034629"),(20036, "ILSVRC2012_val_00022718"),(21815, "ILSVRC2012_val_00043659"),(31039, "ILSVRC2012_val_00043735"),(34554, "ILSVRC2012_val_00008996"),(44344, "ILSVRC2012_val_00010946"),(48005, "ILSVRC2012_val_00034400"),
            (2870, "ILSVRC2012_val_00037086"),(14579, "ILSVRC2012_val_00038221"),(17143, "ILSVRC2012_val_00003387"),(19992, "ILSVRC2012_val_00024678"),(22554, "ILSVRC2012_val_00039315"),(27201, "ILSVRC2012_val_00029860"),(31657, "ILSVRC2012_val_00026993"),(33381, "ILSVRC2012_val_00026527"),(42372, "ILSVRC2012_val_00017991"),(41976, "ILSVRC2012_val_00000887"),
            (5850, "ILSVRC2012_val_00007691"),(14994, "ILSVRC2012_val_00045527"),(13793, "ILSVRC2012_val_00021095"),(24118, "ILSVRC2012_val_00049145"),(22945, "ILSVRC2012_val_00033621"),(24418, "ILSVRC2012_val_00035240"),(39278, "ILSVRC2012_val_00038494"),(33478, "ILSVRC2012_val_00044807"),(47141, "ILSVRC2012_val_00046581"),(47243, "ILSVRC2012_val_00017175"),
            (11936, "ILSVRC2012_val_00015578"),(18624, "ILSVRC2012_val_00001958"),(22679, "ILSVRC2012_val_00038578"),(23748, "ILSVRC2012_val_00003747"),(29739, "ILSVRC2012_val_00033861"),(30376, "ILSVRC2012_val_00025941"),(32801, "ILSVRC2012_val_00040375"),(40351, "ILSVRC2012_val_00007292"),(45533, "ILSVRC2012_val_00018263"),(48709, "ILSVRC2012_val_00003538"),
        ]
        def find_index_by_rank_and_name(rank_1_based: int, name_stub: str) -> Optional[int]:
            idx = int(rank_1_based) - 1
            if 0 <= idx < len(paths_all):
                p = paths_all[idx]
                if p and p != "None" and name_stub in os.path.basename(p):
                    return idx
            for delta in range(1, 6):
                for sign in (-1, 1):
                    j = idx + sign * delta
                    if 0 <= j < len(paths_all):
                        p = paths_all[j]
                        if p and p != "None" and name_stub in os.path.basename(p):
                            return j
            name_stub_lower = name_stub.lower()
            for j, p in enumerate(paths_all):
                if p and p != "None" and name_stub_lower in os.path.basename(p).lower():
                    return j
            return None
        for rank1, stub in explicit:
            j = find_index_by_rank_and_name(rank1, stub)
            if j is None:
                continue
            resolved_indices.append(j)
            resolved_paths.append(paths_all[j])

    if not resolved_indices:
        raise RuntimeError("No explicit images resolved from CSV; check paths and stubs")

    # Compute per-index accuracies
    idx_to_tuple = compute_accuracy_for_indices(C, resolved_indices)

    # Reconstruct per-class mapping preserving order (10 images each)
    per_class: Dict[str, List[int]] = {}

    if args.chosen_dir and os.path.isdir(args.chosen_dir):
        # Use subfolder order as class order; no wnid labels necessary for layout
        subdirs = [os.path.join(args.chosen_dir, d) for d in os.listdir(args.chosen_dir) if os.path.isdir(os.path.join(args.chosen_dir, d))]
        subdirs.sort()
        start = 0
        for i, sd in enumerate(subdirs):
            key = f"row_{i+1}"
            per_class[key] = resolved_indices[start:start+10]
            if len(per_class[key]) < 10:
                per_class[key] = per_class[key] + [None] * (10 - len(per_class[key]))
            start += 10
    else:
        # Map resolved paths back to wnid to form the five canonical rows
        def path_to_wnid_from_path(p: str) -> Optional[str]:
            base = os.path.normpath(p).replace("\\", "/")
            parts = base.split("/")
            for k in range(len(parts) - 1):
                if parts[k].startswith("n") and len(parts[k]) == 9:
                    return parts[k]
            return None
        per_class = {
            "n03187595": [],
            "n03452741": [],
            "n03481172": [],
            "n03637318": [],
            "n02504458": [],
        }
        for j, p in zip(resolved_indices, resolved_paths):
            w = path_to_wnid_from_path(p)
            if w in per_class and len(per_class[w]) < 10:
                per_class[w].append(j)
        for w in list(per_class.keys()):
            lst = per_class[w]
            if len(lst) < 10:
                lst = lst + [None] * (10 - len(lst))
            per_class[w] = lst[:10]

    build_pairs_collage_with_bg(
        per_class_indices=per_class,
        per_index_accuracy=idx_to_tuple,
        all_paths=paths_all,
        out_path=args.out,
        thumb=int(args.thumb),
        pair_gap=int(args.pair_gap),
        dpi=int(args.dpi),
        tiny_font_size=int(args.font_size),
        root_dir=args.root_dir,
    )


if __name__ == "__main__":
    main()


