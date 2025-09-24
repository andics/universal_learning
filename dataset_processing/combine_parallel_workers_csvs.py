from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def default_root_dir(script_path: Path) -> Path:
    # training_gradient_evaluator_single_loss_parallel/output_parallel_8_workers_trial_3 relative to repo root
    # Script lives in dataset_processing/, so go up one directory
    return (script_path.parent.parent / "training_gradient_evaluator_single_loss_parallel" / "output_parallel_8_workers_trial_3").resolve()


def find_model_directories(root_dir: Path) -> List[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    # Sort by name for deterministic output order across models
    return sorted(dirs, key=lambda p: p.name)


_WORKER_FILE_REGEX = re.compile(r"^single_example_results_(\d+)\.csv$")


def select_csvs_for_model(model_dir: Path) -> Tuple[List[Path], bool]:
    """
    Returns a list of CSV files to use for a model and a flag indicating whether a combined file was used.
    Rule:
      - If single_example_results.csv exists, return only that.
      - Otherwise, return worker CSVs sorted by numeric worker id.
    """
    combined = model_dir / "single_example_results.csv"
    if combined.exists() and combined.is_file():
        return [combined], True

    worker_files: List[Tuple[int, Path]] = []
    for entry in model_dir.iterdir():
        if not entry.is_file():
            continue
        m = _WORKER_FILE_REGEX.match(entry.name)
        if not m:
            continue
        worker_idx = int(m.group(1))
        worker_files.append((worker_idx, entry))

    worker_files.sort(key=lambda t: t[0])
    return [p for _, p in worker_files], False


def transform_df(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    required_columns = [
        "path",
        "steps_to_correct",
        "total_loss_sum",
        "final_loss",
        "weight_distance",
        "softmax_wasserstein",
        "grad_mass_wasserstein",
        "global_linear_cka",
        "init_highest_softmax_prob",
        "init_target_softmax_prob",
        "total_steps_to_epsilon",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in input CSV for model '{model_name}': {', '.join(missing)}")

    out = pd.DataFrame({
        "model_name": model_name,
        "image_name": df["path"],
        "steps_to_correct": df["steps_to_correct"],
        "total_loss_sum": df["total_loss_sum"],
        "final_loss": df["final_loss"],
        "weight_distance": df["weight_distance"],
        "softmax_wasserstein": df["softmax_wasserstein"],
        "grad_mass_wasserstein": df["grad_mass_wasserstein"],
        "global_linear_cka": df["global_linear_cka"],
        "init_highest_softmax_prob": df["init_highest_softmax_prob"],
        "init_target_softmax_prob": df["init_target_softmax_prob"],
        "total_steps_to_epsilon": df["total_steps_to_epsilon"],
    })

    # Ensure column order exactly as requested
    desired_order = [
        "model_name",
        "image_name",
        "steps_to_correct",
        "total_loss_sum",
        "final_loss",
        "weight_distance",
        "softmax_wasserstein",
        "grad_mass_wasserstein",
        "global_linear_cka",
        "init_highest_softmax_prob",
        "init_target_softmax_prob",
        "total_steps_to_epsilon",
    ]
    return out[desired_order]


def combine_model_csvs(model_dir: Path) -> pd.DataFrame:
    csvs, used_combined = select_csvs_for_model(model_dir)
    if not csvs:
        return pd.DataFrame()

    parts: List[pd.DataFrame] = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        transformed = transform_df(df, model_dir.name)
        parts.append(transformed)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0, ignore_index=True)


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = default_root_dir(script_path)
    default_out = script_path.parent / "combined_single_example_results_across_models.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Scan model subfolders under the given root, pick combined single_example_results.csv "
            "or worker single_example_results_*.csv in worker order, and write a concatenated CSV "
            "with columns: model_name, image_name, steps_to_correct, total_loss_sum, final_loss, "
            "weight_distance, softmax_wasserstein, grad_mass_wasserstein, global_linear_cka, "
            "init_highest_softmax_prob, init_target_softmax_prob, total_steps_to_epsilon."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(default_root),
        help="Root directory containing per-model subfolders (default: training_gradient_evaluator_single_loss_parallel/output_parallel_8_workers_trial_3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_out),
        help="Output CSV path (default: dataset_processing/combined_single_example_results_across_models.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root_dir).resolve()
    out_path = Path(args.output).resolve()

    model_dirs = find_model_directories(root)
    if not model_dirs:
        print(f"No model subdirectories found under: {root}")
        return 1

    combined_parts: List[pd.DataFrame] = []
    total_rows = 0
    for model_dir in model_dirs:
        try:
            model_df = combine_model_csvs(model_dir)
        except Exception as e:
            print(f"Warning: Skipping model '{model_dir.name}' due to error: {e}")
            continue

        if model_df.empty:
            continue
        combined_parts.append(model_df)
        total_rows += int(model_df.shape[0])

    if not combined_parts:
        print("No data found to combine.")
        return 2

    final_df = pd.concat(combined_parts, axis=0, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    print(f"Wrote combined CSV with {final_df.shape[0]} rows and {final_df.shape[1]} columns to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


