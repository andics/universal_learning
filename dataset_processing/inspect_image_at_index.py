from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "[ObjectNet] Print the image path at a given index from a single-line CSV and "
            "the fraction of models that are correct for that image from an NPY."
        )
    )
    parser.add_argument("--csv", default="./bars/objectnet_examples_ammended.csv", help="Path to ObjectNet examples CSV (single comma-separated line)")
    parser.add_argument("--npy", default="./bars/objectnet.npy", help="Path to ObjectNet correctness/scores NPY [num_models, num_images]")
    parser.add_argument("--index", default=13, type=int, required=False, help="0-based index of the image to inspect")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional threshold to convert float scores into booleans; "
            "if omitted, defaults to 0.5 when NPY is floating-point."
        ),
    )
    return parser.parse_args()


def read_single_line_csv(csv_path: str) -> List[str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        text = f.read().lstrip("\ufeff").strip()
    if not text:
        return []
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    # Keep placeholders like "None"; drop only empty tokens
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p != ""]


def load_correctness_matrix(npy_path: str, threshold: Optional[float]) -> np.ndarray:
    arr = np.load(npy_path, allow_pickle=True)

    # Object arrays may contain None/booleans; normalize to bool numpy array
    if arr.dtype == object:
        if arr.ndim == 1:
            rows = []
            for row in arr.tolist():
                rows.append(np.asarray([False if (x is None) else bool(x) for x in row], dtype=bool))
            M = np.vstack(rows)
            return M
        if arr.ndim == 2:
            num_models, num_images = arr.shape
            M = np.zeros((num_models, num_images), dtype=bool)
            for i in range(num_models):
                for j in range(num_images):
                    x = arr[i, j]
                    M[i, j] = False if (x is None) else bool(x)
            return M
        raise ValueError(f"Unexpected object array shape: {arr.shape}")

    # Floating scores: threshold to booleans
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0)
        thr = 0.5 if threshold is None else float(threshold)
        return (arr > thr)

    # Boolean or other ints: treat nonzero as True
    if arr.dtype == np.bool_:
        return arr
    return (arr != 0)


def main() -> int:
    args = parse_args()

    paths = read_single_line_csv(args.csv)
    if not paths:
        raise SystemExit("CSV appears empty or unreadable")

    idx = int(args.index)
    if idx < 0 or idx >= len(paths):
        raise SystemExit(f"Index out of range: {idx} (CSV length: {len(paths)})")

    M = load_correctness_matrix(args.npy, args.threshold)
    if M.ndim != 2:
        raise SystemExit(f"Expected 2D matrix from NPY, got shape {M.shape}")

    num_models, num_images = int(M.shape[0]), int(M.shape[1])
    if idx >= num_images:
        raise SystemExit(
            f"Index {idx} exceeds NPY columns ({num_images}). Ensure CSV and NPY align."
        )

    col = M[:, idx]
    # Support bool or numeric
    try:
        num_true = int(np.count_nonzero(col))
    except Exception:
        num_true = int(np.sum(col.astype(bool)))
    frac = (float(num_true) / float(num_models)) if num_models > 0 else 0.0

    # Minimal outputs
    print(f"index: {idx}")
    print(f"image_path: {paths[idx]}")
    print(f"num_models: {num_models}")
    print(f"num_true: {num_true}")
    print(f"fraction_correct: {frac:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


