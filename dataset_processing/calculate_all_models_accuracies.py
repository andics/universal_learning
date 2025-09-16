#!/usr/bin/env python3
"""
Compute per-model accuracies from an ImageNet predictions NPY file and write a copy of
imagenet_model_name_mapping.csv with an added 'accuracy' column.

Inputs (CLI args):
  --mapping_csv: Path to mapping CSV with columns [model_in_csv, model_in_timm, parameter_count, url]
  --models_csv:  Path to CSV containing a single row of all model names in rank order
  --scores_npy:  Path to .npy with shape [num_models, 50000], containing values in {None, True, False}

Behavior:
  - For each row in mapping_csv, find its index in models_csv via model_in_csv.
  - Use that index to select the row from scores_npy and compute accuracy = (#True) / 50000.
  - If fewer/more columns are present, divide by the number of columns found.
  - Write an output CSV with the same columns as mapping plus a new 'accuracy' column.

Defaults mirror choose_examples_wrong_in_all_models.py (paths under ./bars/).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate per-model accuracies and append to mapping CSV")
    parser.add_argument("--mapping_csv", default="./bars/imagenet_model_name_mapping.csv", help="Path to imagenet_model_name_mapping.csv")
    parser.add_argument("--models_csv", default="./bars/imagenet_models.csv", help="Path to imagenet_models.csv")
    parser.add_argument("--scores_npy", default="./bars/imagenet.npy", help="Path to imagenet.npy (shape [M, 50000])")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path; default: mapping_csv with _with_accuracy suffix")
    return parser.parse_args()


def read_models_in_rank_order(models_csv_path: str) -> List[str]:
    with open(models_csv_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    models = [token.strip() for token in content.split(",") if token.strip()]
    if not models:
        raise ValueError(f"No model names found in: {models_csv_path}")
    return models


def load_scores_array(scores_npy_path: str) -> np.ndarray:
    # Enable pickle to support object arrays containing None
    arr = np.load(scores_npy_path, allow_pickle=True)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {scores_npy_path}, got shape {arr.shape}")
    return arr


def find_model_index(all_models: List[str], model_name: str) -> int:
    try:
        return all_models.index(model_name)
    except ValueError:
        return -1


def compute_row_accuracy(row: np.ndarray) -> float:
    # Count True; treat None as not-correct and still included in denominator (per spec total=length)
    n = row.shape[0]
    if n <= 0:
        return 0.0
    # For object arrays, np.equal(row, True) works; for bool dtype, row is fine
    try:
        true_mask = np.equal(row, True)
        num_true = int(np.count_nonzero(true_mask))
    except Exception:
        # Fallback: Python iteration (slower, rarely used)
        num_true = sum(1 for v in row.tolist() if v is True)
    return float(num_true) / float(n)


def main() -> int:
    args = parse_args()

    mapping_csv_path = args.mapping_csv
    models_csv_path = args.models_csv
    scores_npy_path = args.scores_npy

    out_path = (
        args.output
        if args.output
        else os.path.splitext(mapping_csv_path)[0] + "_with_accuracy.csv"
    )

    all_models = read_models_in_rank_order(models_csv_path)
    scores = load_scores_array(scores_npy_path)

    if scores.shape[0] < len(all_models):
        print(
            f"Warning: scores rows ({scores.shape[0]}) < number of models ({len(all_models)})",
            file=sys.stderr,
        )

    # Read mapping and produce output rows with accuracy
    with open(mapping_csv_path, "r", encoding="utf-8") as inf, open(out_path, "w", newline="", encoding="utf-8") as outf:
        reader = csv.DictReader(inf)
        fieldnames = list(reader.fieldnames or [])
        if "model_in_csv" not in fieldnames:
            raise ValueError("mapping_csv missing required column: model_in_csv")
        # Ensure 'accuracy' is present as the penultimate column (before the last existing column)
        if len(fieldnames) == 0:
            raise ValueError("mapping_csv appears to have no columns")
        # Remove any existing 'accuracy' to control placement
        fieldnames = [c for c in fieldnames if c != "accuracy"]
        if len(fieldnames) == 1:
            # Only one column exists; put accuracy before the (only) column is not meaningful, so append at end
            fieldnames = fieldnames + ["accuracy"]
        else:
            # Insert as penultimate
            fieldnames = fieldnames[:-1] + ["accuracy"] + fieldnames[-1:]
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            model_name = (row.get("model_in_csv") or "").strip()
            acc = None
            if model_name:
                idx = find_model_index(all_models, model_name)
                if idx >= 0 and idx < scores.shape[0]:
                    model_row = scores[idx]
                    acc = compute_row_accuracy(model_row)
                else:
                    print(f"Warning: Model '{model_name}' not found in models CSV or out of bounds; accuracy left blank", file=sys.stderr)

            out_row = dict(row)
            out_row["accuracy"] = (f"{acc:.6f}" if isinstance(acc, float) else "")
            # csv.DictWriter will order by fieldnames; ensure all keys exist
            for key in fieldnames:
                if key not in out_row:
                    out_row[key] = out_row.get(key, "")
            writer.writerow(out_row)

    print(f"Wrote accuracies to: {os.path.abspath(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


