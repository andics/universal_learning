import argparse
import os
import sys
from typing import List, Tuple


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively traverse a models storage directory and delete .pt files, "
            "keeping only the most recently created ones per folder."
        )
    )
    parser.add_argument(
        "--root",
        default="/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/trained_models",
        help="Path to the root storage directory containing model subfolders",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=2,
        metavar="N",
        help="Number of most recently created .pt files to keep per folder (default: 2)",
    )
    return parser.parse_args()


def validate_inputs(root: str, keep: int) -> None:
    if keep < 0:
        print("--keep must be a non-negative integer", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(root):
        print(f"Path does not exist: {root}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(root):
        print(f"Path is not a directory: {root}", file=sys.stderr)
        sys.exit(2)


def collect_pt_files_in_directory(directory: str) -> List[Tuple[str, float]]:
    pt_files_with_ctime: List[Tuple[str, float]] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if not entry.name.lower().endswith(".pt"):
                    continue
                try:
                    ctime = entry.stat(follow_symlinks=False).st_ctime
                except OSError:
                    # If we can't stat the file, skip it
                    continue
                pt_files_with_ctime.append((entry.path, ctime))
    except OSError as exc:
        print(f"Warning: Could not access directory '{directory}': {exc}", file=sys.stderr)
    return pt_files_with_ctime


def clean_directory(directory: str, keep: int) -> Tuple[int, int]:
    files = collect_pt_files_in_directory(directory)
    if not files:
        return 0, 0

    # Sort by creation time ascending (oldest first)
    files.sort(key=lambda item: item[1])

    num_to_keep = min(keep, len(files)) if keep > 0 else 0
    to_keep = set(path for path, _ in files[-num_to_keep:]) if num_to_keep > 0 else set()
    to_delete = [path for path, _ in files if path not in to_keep]

    kept_count = len(to_keep)
    deleted_count = 0
    for file_path in to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
            print(f"Deleted: {file_path}")
        except OSError as exc:
            print(f"Warning: Failed to delete '{file_path}': {exc}", file=sys.stderr)

    if kept_count:
        print(f"Kept {kept_count} most recent .pt file(s) in '{directory}'.")
    return kept_count, deleted_count


def walk_and_clean(root: str, keep: int) -> Tuple[int, int, int]:
    total_dirs = 0
    total_kept = 0
    total_deleted = 0
    for dirpath, dirnames, filenames in os.walk(root):
        total_dirs += 1
        kept, deleted = clean_directory(dirpath, keep)
        total_kept += kept
        total_deleted += deleted
    return total_dirs, total_kept, total_deleted


def main() -> None:
    args = parse_arguments()
    validate_inputs(args.root, args.keep)
    total_dirs, total_kept, total_deleted = walk_and_clean(args.root, args.keep)
    print(
        (
            f"Scanned {total_dirs} directorie(s). "
            f"Kept {total_kept} .pt file(s). Deleted {total_deleted} .pt file(s)."
        )
    )


if __name__ == "__main__":
    main()


