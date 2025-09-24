from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Optional
import random


def parse_args() -> argparse.Namespace:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bars_dir = os.path.join(project_root, "bars")
    d_csv = os.path.join(bars_dir, "imagenet_examples_ammended.csv")
    d_hier = os.path.join(bars_dir, "imagenet_synset_hierarchy.json")
    d_out = os.path.join(os.path.dirname(__file__), "collage3_placeholder.png")

    p = argparse.ArgumentParser(description="Review dataset: copy all images per WNID into collage_3/<class>/ with rank-prefixed filenames.")
    p.add_argument("--csv", default=d_csv, help="Path to imagenet_examples_ammended.csv")
    p.add_argument("--hier", default=d_hier, help="Path to imagenet_synset_hierarchy.json")
    p.add_argument("--out", default=d_out, help="Output image path (used only to locate collage_3 directory)")
    p.add_argument("--root_dir", type=str, default=None, help="Optional root to prefix non-absolute CSV paths when copying")
    p.add_argument("--seed", type=int, default=1337, help="Random seed used to shuffle class order deterministically")
    return p.parse_args()


def parse_imagenet_examples_csv(csv_path: str) -> List[str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        text = f.read().lstrip("\ufeff").strip()
    if not text:
        return []
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p != ""]


def path_to_wnid(path: str) -> Optional[str]:
    m = re.search(r"/(n\d{8})/", path.replace("\\", "/"))
    return m.group(1) if m else None


def load_hierarchy_labels(hier_path: str) -> Dict[str, str]:
    try:
        with open(hier_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {wnid: meta.get("words", wnid) for wnid, meta in data.items()}
    except Exception:
        return {}


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


def main() -> None:
    args = parse_args()

    paths_all = parse_imagenet_examples_csv(args.csv)
    if not paths_all:
        raise ValueError("CSV appears empty")

    labels = load_hierarchy_labels(args.hier)
    if not labels:
        raise ValueError("Failed to load hierarchy labels; cannot enumerate WNIDs")

    # Build wnid -> indices from CSV once
    wnid_to_indices: Dict[str, List[int]] = {}
    for idx, p in enumerate(paths_all):
        if not p or p == "None":
            continue
        w = path_to_wnid(p)
        if not w:
            continue
        wnid_to_indices.setdefault(w, []).append(idx)

    # Determine collage_3 directory next to --out
    out_dir = os.path.dirname(args.out) or os.getcwd()
    collage_dir = os.path.join(out_dir, "collage_3")
    os.makedirs(collage_dir, exist_ok=True)

    # Global ranking CSV
    global_rows: List[str] = ["wnid,label,rank,image_path"]

    # Iterate all WNIDs found in the hierarchy
    wnids: List[str] = list(labels.keys())
    random.seed(int(args.seed))
    random.shuffle(wnids)
    for wnid in wnids:
        label_text = labels.get(wnid, wnid)
        class_dir = os.path.join(collage_dir, sanitize(label_text))
        if os.path.isdir(class_dir):
            # Skip existing class directory entirely as requested
            continue
        os.makedirs(class_dir, exist_ok=False)

        # Per-class ranking CSV
        class_rows: List[str] = ["rank,image_path"]
        indices = wnid_to_indices.get(wnid, [])
        for idx in indices:
            src = paths_all[idx]
            if not src or src == "None":
                continue
            if not os.path.isabs(src) and args.root_dir:
                src = os.path.join(args.root_dir, src)
            if not os.path.exists(src):
                continue
            rank = idx + 1
            basename = os.path.basename(src)
            dst_name = f"{rank}_{basename}"
            dst = os.path.join(class_dir, dst_name)
            try:
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            except Exception:
                continue
            class_rows.append(f"{rank},{os.path.basename(dst)}")
            global_rows.append(f"{wnid},\"{label_text}\",{rank},\"{src}\"")

        # Write per-class ranking
        try:
            with open(os.path.join(class_dir, "ranking.csv"), 'w', encoding='utf-8') as cf:
                cf.write("\n".join(class_rows))
        except Exception:
            pass

    # Write global ranking
    try:
        with open(os.path.join(collage_dir, "ranking.csv"), 'w', encoding='utf-8') as gf:
            gf.write("\n".join(global_rows))
    except Exception:
        pass

    print(f"Completed dataset review. Output in: {collage_dir}")


if __name__ == "__main__":
    main()


