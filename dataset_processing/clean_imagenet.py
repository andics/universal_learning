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

    # If this is an object array of lists (shape like (num_models,)), convert to 2D bool array
    is_object_vector_of_lists = arr.dtype == object and (arr.ndim == 1)

    if is_object_vector_of_lists:
        list_of_rows = [np.asarray(row, dtype=bool) for row in arr.tolist()]
        matrix = np.vstack(list_of_rows)
        return matrix, True

    # Otherwise ensure boolean dtype
    if arr.dtype != bool:
        arr = arr.astype(bool)

    # Expect shape (num_models, num_images)
    if arr.ndim == 1:
        # Edge case: single model vector
        arr = np.expand_dims(arr, axis=0)

    return arr, False


def compute_keep_indices(correct_matrix: np.ndarray, min_correct: int) -> np.ndarray:

    # Keep images solved by at least `min_correct` models
    num_correct_per_image = np.sum(correct_matrix, axis=0)
    keep_mask = num_correct_per_image >= min_correct
    return np.where(keep_mask)[0]


def save_cleaned_csv(images: List[str], keep_indices: np.ndarray, out_csv_path: str) -> None:

    kept_images = [images[i] for i in keep_indices]
    # Write as a single comma-separated line, matching the described format
    with open(out_csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(kept_images))


def save_cleaned_npy(
    correct_matrix: np.ndarray,
    keep_indices: np.ndarray,
    out_npy_path: str,
    preserve_object_layout: bool,
) -> None:

    filtered = correct_matrix[:, keep_indices]

    if preserve_object_layout:
        # Save back as an object array of Python lists, one list per model
        list_rows = [filtered[i, :].astype(bool).tolist() for i in range(filtered.shape[0])]
        obj_array = np.array(list_rows, dtype=object)
        np.save(out_npy_path, obj_array)
    else:
        np.save(out_npy_path, filtered.astype(bool))


def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.abspath(os.path.join(script_dir, "..", "bars", "imagenet_examples_ammended.csv"))
    default_npy = os.path.abspath(os.path.join(script_dir, "..", "bars", "imagenet.npy"))

    parser = argparse.ArgumentParser(description="Clean ImageNet CSV/NPY by keeping images solved by more than 5 models (configurable).")
    parser.add_argument("--csv", dest="csv_path", default=default_csv, help="Path to source CSV file.")
    parser.add_argument("--npy", dest="npy_path", default=default_npy, help="Path to source NPY file.")
    parser.add_argument(
        "--min-correct",
        dest="min_correct",
        type=int,
        default=150,
        help="Minimum number of models that must be correct to KEEP an image (default: 6).",
    )
    parser.add_argument(
        "--out-csv",
        dest="out_csv",
        default=os.path.join(script_dir, "no_impossible_imagenet_examples_ammended.csv"),
        help="Output CSV path (cleaned).",
    )
    parser.add_argument(
        "--out-npy",
        dest="out_npy",
        default=os.path.join(script_dir, "no_impossible_imagenet.npy"),
        help="Output NPY path (cleaned).",
    )

    args = parser.parse_args()

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

    # Compute indices to keep (>5 models correct by default)
    keep_indices = compute_keep_indices(correct_matrix, min_correct=args.min_correct)

    # Save outputs
    save_cleaned_csv(images, keep_indices, args.out_csv)
    save_cleaned_npy(correct_matrix, keep_indices, args.out_npy, preserve_object_layout)

    removed = num_images_csv - keep_indices.size
    print(
        "\n".join(
            [
                "Cleaning complete:",
                f"- Models: {num_models}",
                f"- Images (original): {num_images_csv}",
                f"- Images kept: {keep_indices.size}",
                f"- Keep threshold (min correct models): {args.min_correct}",
                f"- Images removed: {removed}",
                f"- CSV saved to: {args.out_csv}",
                f"- NPY saved to: {args.out_npy}",
            ]
        )
    )


if __name__ == "__main__":
    main()


