from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List
import numpy as np


def parse_args() -> argparse.Namespace:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bars_dir = os.path.join(project_root, "bars")
    default_csv = os.path.join(bars_dir, "objectnet_examples_ammended.csv")
    # Default ObjectNet images root as provided by the user
    default_images_root = \
        "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/dataset_processing/objectnet-1.0/objectnet-1.0/images/"

    p = argparse.ArgumentParser(
        description=(
            "Review ObjectNet dataset: copy ranked existing images into collage_objectnet/"
            "<class>/ with rank-prefixed filenames (rank is 0-based index in CSV)."
        )
    )
    p.add_argument("--csv", default=default_csv, help="Path to objectnet_examples_ammended.csv")
    p.add_argument(
        "--images_root",
        default=default_images_root,
        help="Path to ObjectNet images root containing class subfolders",
    )
    p.add_argument(
        "--npy",
        default=os.path.join(bars_dir, "objectnet.npy"),
        help="Path to objectnet.npy (models x images correctness or scores)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold to convert float scores to correctness; if None, infer",
    )
    p.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(__file__), "collage_objectnet"),
        help="Output root directory (created if missing)",
    )
    return p.parse_args()


def parse_objectnet_examples_csv(csv_path: str) -> List[str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        text = f.read().lstrip("\ufeff").strip()
    if not text:
        return []
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p != ""]


def sanitize(name: str) -> str:
    return (
        name.replace('/', '_')
        .replace('\\', '_')
        .replace(':', '_')
        .replace('*', '_')
        .replace('?', '_')
        .replace('"', "'")
        .replace('<', '(')
        .replace('>', ')')
        .replace('|', '_')
    )


def build_rank_map(paths_all: List[str], images_root: str) -> Dict[str, int]:
    """
    Build mapping from "class/filename" (normalized with forward slashes) to rank (0-based index in CSV).
    Also indexes by relative path from images_root when applicable.
    """
    rank_map: Dict[str, int] = {}
    norm_images_root = os.path.abspath(images_root).replace('\\', '/').rstrip('/') + '/'
    for idx, raw in enumerate(paths_all):
        if not raw or raw == "None":
            continue
        norm = raw.strip().strip('"').strip("'").replace('\\', '/').replace('//', '/')
        # Absolute variant
        try:
            abs_norm = os.path.abspath(norm).replace('\\', '/')
        except Exception:
            abs_norm = norm
        # Suffix "class/file"
        tail_parts = [p for p in norm.split('/') if p]
        if len(tail_parts) >= 2:
            key_tail2 = f"{tail_parts[-2]}/{tail_parts[-1]}".lower()
            rank_map.setdefault(key_tail2, idx)
        # If under images_root, also index by relpath
        if abs_norm.startswith(norm_images_root):
            rel = abs_norm[len(norm_images_root):]
            rel = '/'.join([p for p in rel.split('/') if p])
            if rel:
                rank_map.setdefault(rel.lower(), idx)
    return rank_map


def list_classes(images_root: str) -> List[str]:
    try:
        entries = os.listdir(images_root)
    except FileNotFoundError:
        return []
    classes: List[str] = []
    for name in entries:
        p = os.path.join(images_root, name)
        if os.path.isdir(p):
            classes.append(name)
    return classes


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


def main() -> None:
    args = parse_args()

    images_root = args.images_root
    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)

    paths_all = parse_objectnet_examples_csv(args.csv)
    if not paths_all:
        raise ValueError("CSV appears empty")

    rank_map = build_rank_map(paths_all, images_root)

    # Load correctness matrix for objectnet
    C = load_objectnet_correctness(args.npy, args.threshold)
    if C.ndim != 2:
        raise ValueError(f"Expected 2D correctness matrix, got {C.shape}")
    num_models = int(C.shape[0])

    classes = list_classes(images_root)
    if not classes:
        raise ValueError("No class subfolders found in images_root")

    global_rows: List[str] = ["class,rank,num_true,num_models,acc_percent,image_path"]

    for cls in classes:
        cls_src_dir = os.path.join(images_root, cls)
        cls_out_dir = os.path.join(out_root, sanitize(cls))
        os.makedirs(cls_out_dir, exist_ok=True)

        class_rows: List[str] = ["rank,num_true,num_models,acc_percent,image_path"]

        try:
            files = os.listdir(cls_src_dir)
        except Exception:
            files = []

        for fname in files:
            src = os.path.join(cls_src_dir, fname)
            if not os.path.isfile(src):
                continue
            key = f"{cls}/{fname}".replace('\\', '/').lower()
            if key not in rank_map:
                continue
            rank = rank_map[key]
            # Compute per-image accuracy over models
            if 0 <= rank < C.shape[1]:
                col = C[:, rank]
                num_true = int(col.sum())
            else:
                num_true = 0
            acc = (float(num_true) / float(num_models)) * 100.0 if num_models > 0 else 0.0
            dst_name = f"{rank}_acc{acc:.1f}_{fname}"
            dst = os.path.join(cls_out_dir, dst_name)
            try:
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            except Exception:
                continue
            class_rows.append(f"{rank},{num_true},{num_models},{acc:.3f},{os.path.basename(dst)}")
            global_rows.append(f"{cls},{rank},{num_true},{num_models},{acc:.3f},\"{src}\"")

        # Write per-class ranking
        try:
            with open(os.path.join(cls_out_dir, "ranking.csv"), 'w', encoding='utf-8') as cf:
                cf.write("\n".join(class_rows))
        except Exception:
            pass

    # Write global ranking
    try:
        with open(os.path.join(out_root, "ranking.csv"), 'w', encoding='utf-8') as gf:
            gf.write("\n".join(global_rows))
    except Exception:
        pass

    print(f"Completed ObjectNet dataset review. Output in: {out_root}")


if __name__ == "__main__":
    main()


