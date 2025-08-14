#!/usr/bin/env python3
"""
Simple CLI to open a .npy file and print summary information and a small preview.

Usage:
  python open_npy.py [path/to/file.npy] [--preview N] [--allow-pickle]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a .npy file and show its contents summary")
    parser.add_argument(
        "path",
        nargs="?",
        default=r"Q:\\Projects\\BMM_school\\Universal_learning\\bars\\imagenet.npy",
        help="Path to the .npy file (default: Q:\\Projects\\BMM_school\\Universal_learning\\bars\\imagenet.npy)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of items/rows/cols to preview (default: 10)",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow loading object arrays saved with pickle (use with caution)",
    )
    return parser.parse_args()


def is_numeric_dtype(dtype: np.dtype) -> bool:
    try:
        return np.issubdtype(dtype, np.number)
    except Exception:
        return False


def print_summary(array: np.ndarray, preview: int) -> None:
    print(f"dtype: {array.dtype}")
    print(f"shape: {array.shape}")
    print(f"ndim: {array.ndim}")
    print(f"size: {array.size}")

    # Basic stats for numeric arrays
    if is_numeric_dtype(array.dtype) and array.size > 0:
        with np.errstate(all="ignore"):
            arr_min = np.nanmin(array)
            arr_max = np.nanmax(array)
            arr_mean = np.nanmean(array)
        print(f"min: {arr_min}")
        print(f"max: {arr_max}")
        print(f"mean: {arr_mean}")
    elif array.dtype == np.bool_:
        true_count = int(np.count_nonzero(array))
        print(f"true_count: {true_count}")
        print(f"false_count: {array.size - true_count}")

    # Preview
    print("preview:")
    try:
        if array.ndim == 0:
            print(array.item())
        elif array.ndim == 1:
            end = min(preview, array.shape[0])
            print(array[:end])
            if end < array.shape[0]:
                print(f"... ({array.shape[0] - end} more elements)")
        elif array.ndim == 2:
            rows = min(preview, array.shape[0])
            cols = min(preview, array.shape[1])
            print(array[:rows, :cols])
            more_rows = array.shape[0] - rows
            more_cols = array.shape[1] - cols
            if more_rows > 0 or more_cols > 0:
                more_parts = []
                if more_rows > 0:
                    more_parts.append(f"{more_rows} more rows")
                if more_cols > 0:
                    more_parts.append(f"{more_cols} more cols")
                print("... (" + ", ".join(more_parts) + ")")
        else:
            # For higher dimensions, show the first few flattened values
            flat_preview = min(preview, array.size)
            sample = np.array(list(array.flat)[:flat_preview])
            print(sample)
            if flat_preview < array.size:
                print(f"... ({array.size - flat_preview} more elements)")
    except Exception as exc:
        print(f"Failed to print preview: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args()
    file_path = args.path

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return 1

    if not file_path.lower().endswith(".npy"):
        print("Error: This utility only supports .npy files.")
        return 1

    try:
        array = np.load(file_path, allow_pickle=args.allow_pickle)
    except Exception as exc:
        print(f"Failed to load file: {exc}")
        return 1

    print(f"path: {os.path.abspath(file_path)}")
    print_summary(array, args.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())


