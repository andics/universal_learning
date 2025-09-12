import os
import re
import argparse
from typing import List, Tuple

import numpy as np


def read_image_list(csv_path: str) -> List[str]:

    with open(csv_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Split by commas and/or newlines, keep order, drop empties
    parts = re.split(r"[\n,]+", text)
    images = [p.strip() for p in parts if p.strip()]
    return images


def load_model_correct_matrix(npy_path: str) -> Tuple[np.ndarray, bool]:

    arr = np.load(npy_path, allow_pickle=True)

    # If this is an object array, handle both shapes:
    # - 1D object array of Python lists (original preserved layout)
    # - 2D object array with potential None entries (masked layout)
    if arr.dtype == object:
        if arr.ndim == 1:
            list_of_rows = []
            for row in arr.tolist():
                bool_row = [False if (x is None) else bool(x) for x in row]
                list_of_rows.append(np.asarray(bool_row, dtype=bool))
            matrix = np.vstack(list_of_rows)
            return matrix, True
        elif arr.ndim == 2:
            num_models, num_images = arr.shape
            matrix = np.zeros((num_models, num_images), dtype=bool)
            for i in range(num_models):
                for j in range(num_images):
                    x = arr[i, j]
                    matrix[i, j] = False if (x is None) else bool(x)
            return matrix, True

    # Otherwise ensure boolean dtype
    if arr.dtype != bool:
        arr = arr.astype(bool)

    # Expect shape (num_models, num_images)
    if arr.ndim == 1:
        # Edge case: single model vector
        arr = np.expand_dims(arr, axis=0)

    return arr, False


def compute_keep_indices(
    correct_matrix: np.ndarray, min_correct: int, min_wrong: int
) -> Tuple[np.ndarray, int, int]:

    # Per-image correct/wrong counts
    num_models = correct_matrix.shape[0]
    num_correct_per_image = np.sum(correct_matrix, axis=0)
    num_wrong_per_image = num_models - num_correct_per_image

    # First criterion: keep images gotten wrong by at least `min_wrong` models
    wrong_keep_mask = num_wrong_per_image >= min_wrong
    # Second criterion: keep images gotten correct by at least `min_correct` models
    correct_keep_mask = num_correct_per_image >= min_correct

    # Apply sequentially: wrong filter, then correct filter
    final_keep_mask = wrong_keep_mask & correct_keep_mask
    keep_indices = np.where(final_keep_mask)[0]

    total_images = correct_matrix.shape[1]
    wrong_removed_count = int(total_images - int(np.sum(wrong_keep_mask)))
    # Removed by correct criterion relative to the original list
    correct_removed_count = int(total_images - int(np.sum(correct_keep_mask)))

    return keep_indices, wrong_removed_count, correct_removed_count


def save_masked_csv(images: List[str], keep_mask: np.ndarray, out_csv_path: str) -> None:

    masked_images = [images[i] if keep_mask[i] else "None" for i in range(len(images))]
    # Write as a single comma-separated line, matching the described format
    with open(out_csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(masked_images))


def save_masked_npy(
    correct_matrix: np.ndarray,
    keep_mask: np.ndarray,
    out_npy_path: str,
    preserve_object_layout: bool,
) -> None:

    num_models, num_images = correct_matrix.shape

    if preserve_object_layout:
        # Save back as a 1D object array of Python lists, one list per model, with None for dropped images
        obj_array = np.empty((num_models,), dtype=object)
        for i in range(num_models):
            row = []
            for j in range(num_images):
                if keep_mask[j]:
                    row.append(bool(correct_matrix[i, j]))
                else:
                    row.append(None)
            obj_array[i] = row
        np.save(out_npy_path, obj_array)
    else:
        # Create object array with None for dropped columns, booleans for kept
        masked = np.empty((num_models, num_images), dtype=object)
        masked[:, :] = None
        if correct_matrix.dtype != bool:
            cm = correct_matrix.astype(bool)
        else:
            cm = correct_matrix
        masked[:, keep_mask] = cm[:, keep_mask]
        np.save(out_npy_path, masked)


def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.abspath(os.path.join(script_dir, "..", "bars", "imagenet_examples_ammended.csv"))
    default_npy = os.path.abspath(os.path.join(script_dir, "..", "bars", "imagenet.npy"))

    parser = argparse.ArgumentParser(description="Clean ImageNet CSV/NPY by keeping images that are wrong by >= min-wrong AND correct by >= min-correct (configurable).")
    parser.add_argument("--csv", dest="csv_path", default=default_csv, help="Path to source CSV file.")
    parser.add_argument("--npy", dest="npy_path", default=default_npy, help="Path to source NPY file.")
    parser.add_argument(
        "--min-correct",
        dest="min_correct",
        type=int,
        default=6,
        help="Minimum number of models that must be correct to KEEP an image.",
    )
    parser.add_argument(
        "--min-wrong",
        dest="min_wrong",
        type=int,
        default=6,
        help="Minimum number of models that must be wrong to KEEP an image.",
    )
    parser.add_argument(
        "--out-csv",
        dest="out_csv",
        default=os.path.join(script_dir, "imagenet_examples_ammended.csv"),
        help="Output CSV path (cleaned).",
    )
    parser.add_argument(
        "--out-npy",
        dest="out_npy",
        default=os.path.join(script_dir, "imagenet.npy"),
        help="Output NPY path (cleaned).",
    )

    args = parser.parse_args()

    # Determine output dirs (actual filenames may be finalized after filtering/counts)
    out_csv_dir = os.path.dirname(args.out_csv) or script_dir
    out_npy_dir = os.path.dirname(args.out_npy) or script_dir

    # Load inputs
    images = read_image_list(args.csv_path)
    correct_matrix, preserve_object_layout = load_model_correct_matrix(args.npy_path)

    num_models = correct_matrix.shape[0]
    num_images_matrix = correct_matrix.shape[1]
    num_images_csv = len(images)

    if num_images_matrix != num_images_csv:
        raise ValueError(
            f"CSV has {num_images_csv} images but NPY has {num_images_matrix} columns. They must match."
        )

    # Compute indices to keep using both criteria
    keep_indices, wrong_removed_count, correct_removed_count = compute_keep_indices(
        correct_matrix,
        min_correct=args.min_correct,
        min_wrong=args.min_wrong,
    )

    # If default output names are used, finalize them with thresholds and removal counts
    default_csv_name_candidates = {"no_impossible_imagenet_examples_ammended.csv", "imagenet_examples_ammended.csv"}
    default_npy_name_candidates = {"no_impossible_imagenet.npy", "imagenet.npy"}

    if os.path.basename(args.out_csv) in default_csv_name_candidates:
        args.out_csv = os.path.join(
            out_csv_dir,
            f"geq{args.min_wrong}wrong_{wrong_removed_count}_geq{args.min_correct}correct_{correct_removed_count}_imagenet_examples_ammended.csv",
        )

    if os.path.basename(args.out_npy) in default_npy_name_candidates:
        args.out_npy = os.path.join(
            out_npy_dir,
            f"geq{args.min_wrong}wrong_geq{args.min_correct}correct_imagenet.npy",
        )

    # Build keep mask and save masked outputs (preserve original length)
    keep_mask = np.zeros(num_images_csv, dtype=bool)
    keep_mask[keep_indices] = True

    save_masked_csv(images, keep_mask, args.out_csv)
    save_masked_npy(correct_matrix, keep_mask, args.out_npy, preserve_object_layout)

    removed = num_images_csv - keep_indices.size
    print(
        "\n".join(
            [
                "Cleaning complete:",
                f"- Models: {num_models}",
                f"- Images (original): {num_images_csv}",
                f"- Images kept: {keep_indices.size}",
                f"- Keep thresholds: min_wrong={args.min_wrong}, min_correct={args.min_correct}",
                f"- Images removed by wrong criterion: {wrong_removed_count}",
                f"- Images removed by correct criterion: {correct_removed_count}",
                f"- Total images removed: {removed}",
                f"- CSV saved to: {args.out_csv}",
                f"- NPY saved to: {args.out_npy}",
            ]
        )
    )


if __name__ == "__main__":
    main()


