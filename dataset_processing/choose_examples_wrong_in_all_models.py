#!/usr/bin/env python3
"""
Compute the set of ImageNet validation images that are predicted incorrectly (False)
by all models listed in an ImageNet model mapping CSV, and write those images to
JSON with their image path and rank (position) in the universal 50,000-example order.

Inputs (CLI args):
  1) imagenet_model_name_mapping.csv  (has columns: model_in_csv, model_in_timm, parameter_count, url)
  2) imagenet_models.csv              (single row, comma-separated list of all model names in rank order)
  3) imagenet.npy                     (numpy array of shape [num_models, 50000] with values in {None, True, False})
  4) imagenet_examples_ammended.csv   (single row, comma-separated list of 50,000 image paths in universal order)

Output:
  - JSON file saved in the same folder as this script, containing objects of the form:
      { "image_rank": <int>, "image_path": <str> }

Notes:
  - The rank of a model is its index in imagenet_models.csv (0-based).
  - The universal image order is defined by imagenet_examples_ammended.csv, which
    aligns with axis-1 of imagenet.npy.

Example:
  python dataset_processing/choose_exmaples_wrong_in_all_models.py \
    bars/imagenet_model_name_mapping.csv \
    bars/imagenet_models.csv \
    bars/imagenet.npy \
    bars/imagenet_examples_ammended.csv \
    --output imagenet_common_false_examples.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List, Dict, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find images that are False across all models listed in the mapping CSV, "
            "using ranks from the models CSV and predictions from the NPY file."
        )
    )
    parser.add_argument("--mapping_csv", default="./bars/imagenet_model_name_mapping_subset_of_8.csv", help="Path to imagenet_model_name_mapping.csv")
    parser.add_argument("--models_csv", default="./bars/imagenet_models.csv", help="Path to imagenet_models.csv")
    parser.add_argument("--scores_npy", default="./bars/geq6wrong_21017_geq6correct_1525_imagenet.npy", help="Path to imagenet.npy (shape [M, 50000])")
    parser.add_argument("--examples_csv", default="./bars/geq6wrong_21017_geq6correct_1525_imagenet_examples_ammended.csv", help="Path to imagenet_examples_ammended.csv")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output JSON filename. Default is 'imagenet_common_false_examples.json' "
            "in the same directory as this script."
        ),
    )
    return parser.parse_args()


def read_models_in_rank_order(models_csv_path: str) -> List[str]:
    with open(models_csv_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # models CSV is a single line of comma-separated model names
    models = [token.strip() for token in content.split(",") if token.strip()]
    if not models:
        raise ValueError(f"No model names found in: {models_csv_path}")
    return models


def read_mapping_models(mapping_csv_path: str) -> List[str]:
    mapping_models: List[str] = []
    with open(mapping_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "model_in_csv" not in reader.fieldnames:
            raise ValueError(
                f"Expected 'model_in_csv' column in mapping file: {mapping_csv_path}"
            )
        for row in reader:
            name = (row.get("model_in_csv") or "").strip()
            if name:
                mapping_models.append(name)
    if not mapping_models:
        raise ValueError(f"No 'model_in_csv' entries found in: {mapping_csv_path}")
    return mapping_models


def read_examples_in_universal_order(examples_csv_path: str) -> List[str]:
    with open(examples_csv_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # examples CSV is a single line of comma-separated image paths
    examples = [token.strip() for token in content.split(",") if token.strip()]
    if not examples:
        raise ValueError(f"No example paths found in: {examples_csv_path}")
    return examples


def resolve_model_indices(
    all_models_in_order: List[str],
    mapping_models: List[str],
) -> List[int]:
    name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(all_models_in_order)}
    resolved_indices: List[int] = []
    missing_models: List[str] = []
    for model_name in mapping_models:
        idx = name_to_index.get(model_name)
        if idx is None:
            missing_models.append(model_name)
        else:
            resolved_indices.append(idx)
    if missing_models:
        print(
            "Warning: The following mapping models were not found in imagenet_models.csv and will be skipped:",
            file=sys.stderr,
        )
        for m in missing_models:
            print(f"  - {m}", file=sys.stderr)
    if not resolved_indices:
        raise ValueError("No mapping models could be resolved against imagenet_models.csv")
    return resolved_indices


def load_scores_array(scores_npy_path: str) -> np.ndarray:
    # We enable allow_pickle=True in case the array is object dtype with None values
    array = np.load(scores_npy_path, allow_pickle=True)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D array in {scores_npy_path}, got shape {array.shape}"
        )
    return array


def compute_common_false_indices(
    scores: np.ndarray,
    selected_model_indices: List[int],
    valid_example_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Subset rows by selected models
    subset = scores[selected_model_indices, :]

    # Vectorized equality to False works for bool and object dtypes
    false_mask = np.equal(subset, False)

    # Common False across all selected models: logical AND across rows (axis=0)
    common_false_mask = np.all(false_mask, axis=0)

    # Exclude examples that are None in the examples CSV via a validity mask
    if valid_example_mask is not None:
        if valid_example_mask.shape[0] != common_false_mask.shape[0]:
            raise ValueError(
                "valid_example_mask length does not match scores columns: "
                f"{valid_example_mask.shape[0]} vs {common_false_mask.shape[0]}"
            )
        common_false_mask = np.logical_and(common_false_mask, valid_example_mask)

    # Indices of images that are False for all selected models
    common_false_indices = np.flatnonzero(common_false_mask)
    return common_false_indices


def main(argv: List[str] | None = None) -> int:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = (
        args.output
        if args.output
        else os.path.join(script_dir, "imagenet_common_false_examples.json")
    )

    # Load inputs
    all_models_in_order = read_models_in_rank_order(args.models_csv)
    mapping_models = read_mapping_models(args.mapping_csv)
    scores = load_scores_array(args.scores_npy)
    example_paths = read_examples_in_universal_order(args.examples_csv)

    # Validate alignment between scores and examples
    if scores.shape[1] != len(example_paths):
        raise ValueError(
            "Mismatch between number of examples and scores: "
            f"scores has {scores.shape[1]} columns, examples has {len(example_paths)} entries"
        )

    # Resolve selected model indices
    selected_model_indices = resolve_model_indices(all_models_in_order, mapping_models)

    # Build mask of valid examples (exclude entries that are exactly the string 'None')
    valid_example_mask = np.array([p != "None" for p in example_paths], dtype=bool)

    # Compute common False indices across selected models, restricted to valid examples
    common_false_indices = compute_common_false_indices(
        scores, selected_model_indices, valid_example_mask
    )

    # Build output records
    results = [
        {"image_rank": int(idx), "image_path": example_paths[int(idx)]}
        for idx in common_false_indices
    ]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print a brief summary
    print(
        f"Processed {len(selected_model_indices)} mapped models "
        f"out of {len(all_models_in_order)} total models."
    )
    print(f"Common false images: {len(results)}")
    print(f"Output written to: {os.path.abspath(output_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


