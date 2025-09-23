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
    d_out = os.path.join(os.path.dirname(__file__), "accuracy_image_collage.png")

    p = argparse.ArgumentParser(description="Build a simple collage of selected images with per-image accuracy from imagenet.npy.")
    p.add_argument("--csv", default=d_csv, help="Path to imagenet_examples_ammended.csv (single row CSV of 50k paths)")
    p.add_argument("--npy", default=d_npy, help="Path to imagenet.npy (shape [num_models, num_images])")
    p.add_argument("--out", default=d_out, help="Output collage image path")
    p.add_argument("--root_dir", type=str, default=None, help="Optional root dir to prefix to CSV paths to load images")
    p.add_argument("--thumb", type=int, default=160, help="Thumbnail size (square side in px)")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI")
    p.add_argument("--cols", type=int, default=10, help="Number of columns in collage")
    p.add_argument("--rows", type=int, default=5, help="Number of rows in collage")
    p.add_argument("--font_size", type=int, default=8, help="Font size for accuracy text")
    p.add_argument("--threshold", type=float, default=None, help="Optional threshold to convert float scores to correctness; if None, infer from dtype")
    p.add_argument("--wnids", type=str, default="n03187595,n03452741,n03481172,n03637318,n02504458", help="Comma-separated wnids; images will be drawn in listed order of each wnid's mapping list")
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


def open_image(path: str, thumb: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((thumb, thumb), Image.LANCZOS)
        out = Image.new("RGB", (thumb, thumb), (0, 0, 0))
        # center paste
        w, h = im.size
        x = (thumb - w) // 2
        y = (thumb - h) // 2
        out.paste(im, (x, y))
        return out


def build_simple_collage(
    image_paths: List[str],
    accuracies: List[Tuple[int, int, float]],
    out_path: str,
    thumb: int,
    cols: int,
    rows: int,
    dpi: int,
    font_size: int,
) -> None:
    assert len(image_paths) == len(accuracies)
    total = len(image_paths)
    cols = max(1, cols)
    rows = max(1, rows)
    fig_w = cols * (thumb / dpi) * 1.05
    fig_h = rows * (thumb / dpi) * 1.25
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.axis('off')
        if idx >= total:
            continue
        path = image_paths[idx]
        try:
            im = open_image(path, thumb)
            ax.imshow(im)
        except Exception:
            # draw empty tile
            blank = Image.new("RGB", (thumb, thumb), (30, 30, 30))
            ax.imshow(blank)
        num_true, num_models, acc = accuracies[idx]
        ax.set_title(f"{num_true}/{num_models} ({acc*100:.1f}%)", fontsize=font_size)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
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

    # Define explicit mapping lists: follow the same examples but simplified to just filenames
    # We will locate each by searching around the provided rank or by exact filename match.
    # The ranks below are 1-based; convert to 0-based.
    explicit = [
        # n03187595 Dial phone
        (1203, "ILSVRC2012_val_00034672"),
        (1903, "ILSVRC2012_val_00022385"),
        (10502, "ILSVRC2012_val_00029045"),
        (13362, "ILSVRC2012_val_00013729"),
        (22345, "ILSVRC2012_val_00016388"),
        (22559, "ILSVRC2012_val_00029370"),
        (31523, "ILSVRC2012_val_00017123"),
        (36050, "ILSVRC2012_val_00033592"),
        (42722, "ILSVRC2012_val_00016249"),
        (43886, "ILSVRC2012_val_00000137"),
        # n03452741 Grand piano
        (2581, "ILSVRC2012_val_00021661"),
        (6089, "ILSVRC2012_val_00007940"),
        (13804, "ILSVRC2012_val_00013511"),
        (13855, "ILSVRC2012_val_00034629"),
        (20036, "ILSVRC2012_val_00022718"),
        (21815, "ILSVRC2012_val_00043659"),
        (31039, "ILSVRC2012_val_00043735"),
        (34554, "ILSVRC2012_val_00008996"),
        (44344, "ILSVRC2012_val_00010946"),
        (48005, "ILSVRC2012_val_00034400"),
        # n03481172 Hammer
        (2870, "ILSVRC2012_val_00037086"),
        (14579, "ILSVRC2012_val_00038221"),
        (17143, "ILSVRC2012_val_00003387"),
        (19992, "ILSVRC2012_val_00024678"),
        (22554, "ILSVRC2012_val_00039315"),
        (27201, "ILSVRC2012_val_00029860"),
        (31657, "ILSVRC2012_val_00026993"),
        (33381, "ILSVRC2012_val_00026527"),
        (42372, "ILSVRC2012_val_00017991"),
        (41976, "ILSVRC2012_val_00000887"),
        # n03637318 Lampshade
        (5850, "ILSVRC2012_val_00007691"),
        (14994, "ILSVRC2012_val_00045527"),
        (13793, "ILSVRC2012_val_00021095"),
        (24118, "ILSVRC2012_val_00049145"),
        (22945, "ILSVRC2012_val_00033621"),
        (24418, "ILSVRC2012_val_00035240"),
        (39278, "ILSVRC2012_val_00038494"),
        (33478, "ILSVRC2012_val_00044807"),
        (47141, "ILSVRC2012_val_00046581"),
        (47243, "ILSVRC2012_val_00017175"),
        # n02504458 Elephant
        (11936, "ILSVRC2012_val_00015578"),
        (18624, "ILSVRC2012_val_00001958"),
        (22679, "ILSVRC2012_val_00038578"),
        (23748, "ILSVRC2012_val_00003747"),
        (29739, "ILSVRC2012_val_00033861"),
        (30376, "ILSVRC2012_val_00025941"),
        (32801, "ILSVRC2012_val_00040375"),
        (40351, "ILSVRC2012_val_00007292"),
        (45533, "ILSVRC2012_val_00018263"),
        (48709, "ILSVRC2012_val_00003538"),
    ]

    # Utility to find index by nearby rank and filename stub
    def find_index_by_rank_and_name(rank_1_based: int, name_stub: str) -> Optional[int]:
        idx = int(rank_1_based) - 1
        # First, check exact slot
        if 0 <= idx < len(paths_all):
            p = paths_all[idx]
            if p and p != "None" and name_stub in os.path.basename(p):
                return idx
        # Fallback: scan neighborhood
        for delta in range(1, 6):
            for sign in (-1, 1):
                j = idx + sign * delta
                if 0 <= j < len(paths_all):
                    p = paths_all[j]
                    if p and p != "None" and name_stub in os.path.basename(p):
                        return j
        # Last resort: full scan by basename contains stub
        name_stub_lower = name_stub.lower()
        for j, p in enumerate(paths_all):
            if p and p != "None" and name_stub_lower in os.path.basename(p).lower():
                return j
        return None

    resolved_indices: List[int] = []
    resolved_paths: List[str] = []
    for rank1, stub in explicit:
        j = find_index_by_rank_and_name(rank1, stub)
        if j is None:
            continue
        resolved_indices.append(j)
        resolved_paths.append(paths_all[j])

    if not resolved_indices:
        raise RuntimeError("No explicit images resolved from CSV; check paths and stubs")

    # Compute accuracies
    idx_to_tuple = compute_accuracy_for_indices(C, resolved_indices)
    acc_list: List[Tuple[int, int, float]] = [idx_to_tuple[j] for j in resolved_indices]

    # Build file paths for opening
    def to_full_path(p: str) -> str:
        return os.path.join(args.root_dir, p.lstrip("/\\")) if args.root_dir else p

    full_image_paths = [to_full_path(p) for p in resolved_paths]

    # Compute grid size based on requested rows/cols and number of images
    cols = max(1, int(args.cols))
    rows = max(1, int(args.rows))
    capacity = rows * cols
    if len(full_image_paths) < capacity:
        # trim rows to fit exactly if there are fewer images
        rows = max(1, (len(full_image_paths) + cols - 1) // cols)

    build_simple_collage(
        image_paths=full_image_paths,
        accuracies=acc_list,
        out_path=args.out,
        thumb=int(args.thumb),
        cols=cols,
        rows=rows,
        dpi=int(args.dpi),
        font_size=int(args.font_size),
    )


if __name__ == "__main__":
    main()


