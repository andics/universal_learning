from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Reuse helper logic by importing from the main script
from visualizations.create_difficulty_image_spectrum_figure import (
    parse_imagenet_examples_csv,
    path_to_wnid,
    load_hierarchy_labels,
    continuous_colormap,
    linear_gradient_rgb,
    build_and_save_collage,
)


def parse_args() -> argparse.Namespace:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bars_dir = os.path.join(project_root, "bars")
    d_csv = os.path.join(bars_dir, "imagenet_examples_ammended.csv")
    d_hier = os.path.join(bars_dir, "imagenet_synset_hierarchy.json")
    d_out = os.path.join(os.path.dirname(__file__), "difficulty_spectrum_third.png")

    p = argparse.ArgumentParser(description="Generate only Collage 3 (explicit selections and variants).")
    p.add_argument("--csv", default=d_csv)
    p.add_argument("--hier", default=d_hier)
    p.add_argument("--out", default=d_out)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--classes", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--thumb", type=int, default=160)
    p.add_argument("--dpi", type=int, default=350)
    p.add_argument("--pair_gap", type=int, default=12)
    p.add_argument("--root_dir", type=str, default=None)
    p.add_argument("--copy_images", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is None:
        random.seed()
    else:
        random.seed(args.seed)

    paths_all = parse_imagenet_examples_csv(args.csv)
    if len(paths_all) <= 0:
        raise ValueError("CSV appears empty")
    labels = load_hierarchy_labels(args.hier)

    # Explicit selections for collage 3
    def find_index_by_rank_and_name(rank: int, name_stub: str) -> Optional[int]:
        idx = int(rank) - 1
        if 0 <= idx < len(paths_all):
            path = paths_all[idx]
            if path and path != "None" and name_stub in os.path.basename(path):
                return idx
        for delta in range(1, 5):
            for sign in (-1, 1):
                j = idx + sign * delta
                if 0 <= j < len(paths_all):
                    path = paths_all[j]
                    if path and path != "None" and name_stub in os.path.basename(path):
                        return j
        return None

    explicit_map: Dict[str, List[int]] = {}
    # Dial phone n03187595
    explicit_map["n03187595"] = [
        find_index_by_rank_and_name(1203, "ILSVRC2012_val_00034672"),
        find_index_by_rank_and_name(1903, "ILSVRC2012_val_00022385"),
        find_index_by_rank_and_name(10502, "ILSVRC2012_val_00029045"),
        find_index_by_rank_and_name(13362, "ILSVRC2012_val_00013729"),
        find_index_by_rank_and_name(22345, "ILSVRC2012_val_00016388"),
        find_index_by_rank_and_name(22559, "ILSVRC2012_val_00029370"),
        find_index_by_rank_and_name(31523, "ILSVRC2012_val_00017123"),
        find_index_by_rank_and_name(36050, "ILSVRC2012_val_00033592"),
        find_index_by_rank_and_name(42722, "ILSVRC2012_val_00016249"),
        find_index_by_rank_and_name(43886, "ILSVRC2012_val_00000137"),
    ]
    # Grand piano n03452741
    explicit_map["n03452741"] = [
        find_index_by_rank_and_name(2581, "ILSVRC2012_val_00021661"),
        find_index_by_rank_and_name(6089, "ILSVRC2012_val_00007940"),
        find_index_by_rank_and_name(13804, "ILSVRC2012_val_00013511"),
        find_index_by_rank_and_name(13855, "ILSVRC2012_val_00034629"),
        find_index_by_rank_and_name(20036, "ILSVRC2012_val_00022718"),
        find_index_by_rank_and_name(21815, "ILSVRC2012_val_00043659"),
        find_index_by_rank_and_name(31039, "ILSVRC2012_val_00043735"),
        find_index_by_rank_and_name(34554, "ILSVRC2012_val_00008996"),
        find_index_by_rank_and_name(44344, "ILSVRC2012_val_00010946"),
        find_index_by_rank_and_name(48005, "ILSVRC2012_val_00034400"),
    ]
    # Hammer n03481172
    explicit_map["n03481172"] = [
        find_index_by_rank_and_name(2870, "ILSVRC2012_val_00037086"),
        find_index_by_rank_and_name(14579, "ILSVRC2012_val_00038221"),
        find_index_by_rank_and_name(17143, "ILSVRC2012_val_00003387"),
        find_index_by_rank_and_name(19992, "ILSVRC2012_val_00024678"),
        find_index_by_rank_and_name(22554, "ILSVRC2012_val_00039315"),
        find_index_by_rank_and_name(27201, "ILSVRC2012_val_00029860"),
        find_index_by_rank_and_name(31657, "ILSVRC2012_val_00026993"),
        find_index_by_rank_and_name(33381, "ILSVRC2012_val_00026527"),
        find_index_by_rank_and_name(42372, "ILSVRC2012_val_00017991"),
        find_index_by_rank_and_name(41976, "ILSVRC2012_val_00000887"),
    ]
    # Lampshade n03637318
    explicit_map["n03637318"] = [
        find_index_by_rank_and_name(5850, "ILSVRC2012_val_00007691"),
        find_index_by_rank_and_name(14994, "ILSVRC2012_val_00045527"),
        find_index_by_rank_and_name(13793, "ILSVRC2012_val_00021095"),
        find_index_by_rank_and_name(24118, "ILSVRC2012_val_00049145"),
        find_index_by_rank_and_name(22945, "ILSVRC2012_val_00033621"),
        find_index_by_rank_and_name(24418, "ILSVRC2012_val_00035240"),
        find_index_by_rank_and_name(39278, "ILSVRC2012_val_00038494"),
        find_index_by_rank_and_name(33478, "ILSVRC2012_val_00044807"),
        find_index_by_rank_and_name(47141, "ILSVRC2012_val_00046581"),
        find_index_by_rank_and_name(47243, "ILSVRC2012_val_00017175"),
    ]
    # Elephant n02504458
    explicit_map["n02504458"] = [
        find_index_by_rank_and_name(11936, "ILSVRC2012_val_00015578"),
        find_index_by_rank_and_name(18624, "ILSVRC2012_val_00001958"),
        find_index_by_rank_and_name(22679, "ILSVRC2012_val_00038578"),
        find_index_by_rank_and_name(23748, "ILSVRC2012_val_00003747"),
        find_index_by_rank_and_name(29739, "ILSVRC2012_val_00033861"),
        find_index_by_rank_and_name(30376, "ILSVRC2012_val_00025941"),
        find_index_by_rank_and_name(32801, "ILSVRC2012_val_00040375"),
        find_index_by_rank_and_name(40351, "ILSVRC2012_val_00007292"),
        find_index_by_rank_and_name(45533, "ILSVRC2012_val_00018263"),
        find_index_by_rank_and_name(48709, "ILSVRC2012_val_00003538"),
    ]

    third_wnids = "n03187595,n03452741,n03481172,n03637318,n02504458"
    root, ext = os.path.splitext(args.out)
    base_third = f"{root}"

    variations = [
        ("cmap", "magma", 0.10, 0.60, 0.70),
        ("cmap", "cividis", 0.20, 0.80, 0.70),
        ("cmap", "viridis", 0.30, 0.80, 0.65),
        ("custom_yellow_blue", None, 0.0, 1.0, 0.75),
        ("custom_yellow_blue_fixed", None, 0.0, 1.0, 0.75),
    ]

    # Generate all variants
    for vidx, (mode, cmap_name, vmin, vmax, alpha) in enumerate(variations):
        setattr(args, "_bg_mode", mode)
        if cmap_name is not None:
            setattr(args, "_bg_cmap", cmap_name)
        setattr(args, "_bg_vmin", vmin)
        setattr(args, "_bg_vmax", vmax)
        setattr(args, "_bg_alpha", alpha)
        out_path = f"{base_third}_{vidx}{ext or '.png'}"
        build_and_save_collage(paths_all, labels, args, third_wnids, out_path, collage_index=3, explicit_picks_by_wnid=explicit_map)


if __name__ == "__main__":
    main()


