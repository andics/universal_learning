from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    d_npy = os.path.join(project_root, "bars", "objectnet.npy")
    d_out = os.path.join(os.path.dirname(__file__), "difficulty_objectnet_collage.png")
    d_chosen = os.path.join(os.path.dirname(__file__), "collage_objectnet_chosen")

    p = argparse.ArgumentParser(
        description=(
            "Build a collage like create_accuracy_image_collage_simple.py, "
            "but for ObjectNet images in collage_objectnet_chosen/ with filenames "
            "formatted as '<rank>_acc<percent>_*.ext'."
        )
    )
    p.add_argument("--chosen_dir", default=d_chosen, help="Root folder containing subfolders (rows) of chosen images")
    p.add_argument("--npy", default=d_npy, help="Path to objectnet.npy (shape [num_models, num_images])")
    p.add_argument("--threshold", type=float, default=None, help="Optional threshold for float scores -> correctness; if None, infer")
    p.add_argument("--out", default=d_out, help="Output collage image path (PNG); PDF copy is also saved")
    p.add_argument("--thumb", type=int, default=160, help="Thumbnail size (square side in px)")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI")
    p.add_argument("--pair_gap", type=int, default=12, help="Gap between image pairs (px)")
    p.add_argument("--font_size", type=int, default=6, help="Tiny font size for accuracy text")
    return p.parse_args()


def load_objectnet_correctness(npy_path: str, threshold: float | None) -> np.ndarray:
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


def parse_rank_and_acc_from_filename(filename: str) -> Optional[Tuple[int, float]]:
    # Expected like: "10105_acc14.5_*.png" or similar
    base = os.path.basename(filename)
    m = re.match(r"^(\d+)_acc([0-9]+(?:\.[0-9]+)?)_", base)
    if not m:
        return None
    try:
        rank = int(m.group(1))
        acc_percent = float(m.group(2))
        return rank, acc_percent
    except Exception:
        return None


def compute_accuracy_for_ranks(correct_matrix: np.ndarray, ranks: List[int]) -> Dict[int, Tuple[int, int, float]]:
    result: Dict[int, Tuple[int, int, float]] = {}
    if correct_matrix.ndim != 2:
        raise ValueError(f"Expected 2D correctness matrix, got {correct_matrix.shape}")
    num_models = int(correct_matrix.shape[0])
    for j in ranks:
        if j < 0 or j >= correct_matrix.shape[1]:
            result[j] = (0, num_models, 0.0)
            continue
        col = correct_matrix[:, j]
        num_true = int(col.sum())
        acc = float(num_true) / float(num_models) if num_models > 0 else 0.0
        result[j] = (num_true, num_models, acc)
    return result


def build_pairs_collage_with_bg_files(
    per_class_entries: Dict[str, List[Optional[Tuple[str, int, float]]]],
    per_rank_accuracy: Dict[int, Tuple[int, int, float]],
    out_path: str,
    thumb: int,
    pair_gap: int,
    dpi: int,
    tiny_font_size: int,
) -> None:
    # Match layout and style of the imagenet collage: rows of 5 pairs (10 images) per class,
    # gradient background, transparent figure, black border around pairs, tiny text.
    from PIL import ImageDraw, ImageFont

    chosen_rows = list(per_class_entries.keys())
    rows = len(chosen_rows)
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

    # Background gradient matching custom_yellow_blue_fixed, with white in the middle
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

    # left (253,231,36) #FDE724, middle white, right (68,1,84) #440154
    gradient = three_color_gradient_rgb(2000, (253, 231, 36), (255, 255, 255), (68, 1, 84))
    bg_ax.imshow(gradient, aspect="auto", extent=[0, 1, 0, 1], alpha=0.75)
    bg_ax.set_axis_off()

    # Ensure content axes above background
    try:
        for r in range(len(height_ratios)):
            for c in range(cols_total):
                axes[r, c].set_zorder(10)
    except Exception:
        pass

    # Font for tiny text: scalable font at ~2x base size
    def load_font_scaled(base_px: int) -> Optional[ImageFont.ImageFont]:
        size = max(1, int(round(base_px * 2.0)))
        try:
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

    def compose_pair(left_entry: Optional[Tuple[str, int, float]], right_entry: Optional[Tuple[str, int, float]]) -> Image.Image:
        left_img = Image.new("RGBA", single_size, (0, 0, 0, 255)) if left_entry is None else open_image_rgba_centered(left_entry[0], single_size)
        right_img = Image.new("RGBA", single_size, (0, 0, 0, 255)) if right_entry is None else open_image_rgba_centered(right_entry[0], single_size)
        pair_img = Image.new("RGBA", pair_size, (0, 0, 0, 0))
        pair_img.paste(left_img, (0, 0))
        pair_img.paste(right_img, (tile, 0))

        # Prepare drawing context and draw border first
        from PIL import ImageDraw
        draw = ImageDraw.Draw(pair_img)
        border_px = max(1, int(round(tile * 0.02)))
        for k in range(border_px):
            draw.rectangle([k, k, pair_size[0]-1-k, pair_size[1]-1-k], outline=(0, 0, 0, 255))

        # draw tiny accuracy text bottom-centered on each half
        def text_for(entry: Optional[Tuple[str, int, float]]) -> str:
            if entry is None:
                return ""
            _path, rank, _acc_prefix = entry
            nt, nm, acc = per_rank_accuracy.get(int(rank), (0, 0, 0.0))
            return f"{nt}/{nm} ({acc*100:.1f}%)"

        left_text = text_for(left_entry)
        right_text = text_for(right_entry)

        # bottom-centered positions
        padding_px = max(2, int(round(tile * 0.05)))

        def draw_outlined_text_bottom_center(x_center: int, text: str) -> None:
            if not text:
                return
            try:
                w, h = draw.textsize(text, font=font) if font else draw.textsize(text)
            except Exception:
                w, h = (len(text) * 3, 6)
            # shift 30% further left relative to centered position, then clamp
            x = int(round(x_center - (w / 2.0) - 0.61 * w))
            x = max(border_px + 1, x)
            y = int(tile - border_px - padding_px - h)
            y = max(border_px + 1, y)
            # outline
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0, 255), font=font)
            draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

        draw_outlined_text_bottom_center(tile // 2, left_text)
        draw_outlined_text_bottom_center(tile + (tile // 2), right_text)

        return pair_img

    # place pairs
    for r_idx, key in enumerate(chosen_rows):
        axis_row = 1 + r_idx * 2 if rows > 1 else 1
        picks = per_class_entries.get(key, [])
        for p in range(num_pairs):
            left_bin = p * 2
            right_bin = left_bin + 1
            left_entry = picks[left_bin] if left_bin < len(picks) else None
            right_entry = picks[right_bin] if right_bin < len(picks) else None
            col_idx = pair_to_col_index(p)
            ax = axes[axis_row, col_idx]
            ax.set_axis_off()
            pair_img = compose_pair(left_entry, right_entry)
            ax.set_facecolor("none")
            ax.imshow(pair_img, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
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

    if not args.chosen_dir or not os.path.isdir(args.chosen_dir):
        raise ValueError(f"Chosen dir not found: {args.chosen_dir}")

    # Enumerate subfolders as rows and parse rank/accuracy from filenames
    subdirs = [os.path.join(args.chosen_dir, d) for d in os.listdir(args.chosen_dir) if os.path.isdir(os.path.join(args.chosen_dir, d))]
    subdirs.sort()

    # Gather all ranks for accuracy computation
    unique_ranks: List[int] = []
    per_class_entries: Dict[str, List[Optional[Tuple[str, int, float]]]] = {}

    for i, sd in enumerate(subdirs):
        files = [f for f in os.listdir(sd) if not f.startswith('.')]
        parsed: List[Tuple[str, int, float]] = []
        for fname in files:
            res = parse_rank_and_acc_from_filename(fname)
            if res is None:
                continue
            rank, acc_percent = res
            full_path = os.path.join(sd, fname)
            parsed.append((full_path, rank, acc_percent))
            unique_ranks.append(rank)
        # sort by rank ascending and keep first 10 (expected)
        parsed.sort(key=lambda t: t[1])
        row_entries: List[Optional[Tuple[str, int, float]]] = parsed[:10]
        if len(row_entries) < 10:
            row_entries = row_entries + [None] * (10 - len(row_entries))
        per_class_entries[f"row_{i+1}"] = row_entries

    if not unique_ranks:
        raise RuntimeError("No images with rank/acc prefixes found in chosen_dir")

    # Compute per-rank accuracies from npy
    C = load_objectnet_correctness(args.npy, args.threshold)
    per_rank_acc = compute_accuracy_for_ranks(C, sorted(set(unique_ranks)))

    # Sanity-check provided accuracies in filenames vs computed
    mismatches: List[Tuple[str, float, float]] = []
    num_models = int(C.shape[0]) if C.ndim == 2 else 0
    for row_entries in per_class_entries.values():
        for entry in row_entries:
            if entry is None:
                continue
            path, rank, acc_prefix = entry
            nt, nm, acc = per_rank_acc.get(rank, (0, num_models, 0.0))
            acc_pct = acc * 100.0
            if abs(acc_pct - acc_prefix) > 0.51:
                mismatches.append((path, acc_prefix, acc_pct))
    if mismatches:
        print(f"Warning: {len(mismatches)} filename accuracy values differ from computed values (>|0.51|%). Showing first 10:")
        for path, got, exp in mismatches[:10]:
            print(f"  {os.path.basename(path)}: filename acc={got:.2f}%, computed acc={exp:.2f}%")

    # Render collage
    build_pairs_collage_with_bg_files(
        per_class_entries=per_class_entries,
        per_rank_accuracy=per_rank_acc,
        out_path=args.out,
        thumb=int(args.thumb),
        pair_gap=int(args.pair_gap),
        dpi=int(args.dpi),
        tiny_font_size=int(args.font_size),
    )

    print(f"Saved collage to: {args.out}")


if __name__ == "__main__":
    main()


