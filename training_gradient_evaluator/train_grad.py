import argparse
import csv
import os, sys
from typing import Dict, List, Tuple
from pathlib import Path

# Ensure working directory and sys.path point to the Programming root so package imports resolve
try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
    # Optional: remove paths that might conflict in some environments
    try:
        sys.path.remove('/workspace/object_detection')
    except Exception:
        pass
    if path_main not in sys.path:
        sys.path.append(path_main)
    os.chdir(path_main)
    print(f"Set working directory and sys.path to: {path_main}")
except Exception as _e:
    print("Warning: Failed to adjust working directory/sys.path:", _e)

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm
from timm.data import resolve_model_data_config, create_transform

from training_gradient_evaluator.data import ImageNetWrongExamplesDataset, read_imagenet_paths, read_synset_to_index


def filter_existing_indices(paths: List[str], indices: List[int], root_dir: str | None) -> List[int]:
    kept: List[int] = []
    for idx in indices:
        p = paths[idx]
        full = os.path.join(root_dir, p) if root_dir and not os.path.isabs(p) else p
        if os.path.exists(full):
            kept.append(idx)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Train timm mobilenetv3 on only the images it originally got wrong; log per-step RIGHT/WRONG and first-correct step per example.")
    parser.add_argument("--model_name", type=str, default="mobilenetv3_small_050.lamb_in1k")
    parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "imagenet.npy"))
    parser.add_argument("--examples_csv", type=str, default=os.path.join("bars", "imagenet_examples_ammended.csv"))
    parser.add_argument("--mapping_txt", type=str, default=os.path.join("image_difficulty_classifier", "imagenet_class_name_mapping.txt"))
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--mask_row_index", type=int, default=1022, help="Row for mobilenetv3_small_050.lamb_in1k in imagenet.npy")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=os.path.join("training_gradient_evaluator", "outputs"))
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Paths and labels
    paths = read_imagenet_paths(args.examples_csv)
    if not paths:
        raise RuntimeError(f"No image paths found in {args.examples_csv}")
    synset_to_index = read_synset_to_index(args.mapping_txt)

    # Load original correctness mask and select wrong examples for the specified model row
    mask = np.load(args.bars_npy)
    if mask.ndim != 2 or args.mask_row_index < 0 or args.mask_row_index >= mask.shape[0]:
        raise ValueError(f"Unexpected mask shape {mask.shape} or bad row {args.mask_row_index}")
    correct_mask = mask[args.mask_row_index].astype(bool)  # True means originally correct
    wrong_mask = ~correct_mask
    wrong_indices = np.nonzero(wrong_mask)[0].tolist()
    wrong_indices = filter_existing_indices(paths, wrong_indices, args.root_dir)
    if not wrong_indices:
        raise RuntimeError("No existing images among wrong examples.")
    print(f"Training on {len(wrong_indices)} originally-wrong examples.")

    # Build model and transforms from timm config
    model = timm.create_model(args.model_name, pretrained=True, num_classes=1000)
    model = model.to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    data_cfg = resolve_model_data_config(model)
    train_tfms = create_transform(**data_cfg, is_training=True)
    eval_tfms = create_transform(**data_cfg, is_training=False)

    # Dataset and loaders
    train_ds = ImageNetWrongExamplesDataset(paths, wrong_indices, synset_to_index, transform=train_tfms, root_dir=args.root_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Optimizer and AMP
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = None if args.no_amp or device.type != "cuda" else torch.cuda.amp.GradScaler()

    # Prepare first-correct tracking for each example path; only for those in training set
    # Use resolved full paths as keys
    def resolve_full(p: str) -> str:
        return os.path.join(args.root_dir, p) if args.root_dir and not os.path.isabs(p) else p

    train_paths_full = [resolve_full(paths[i]) for i in wrong_indices]
    first_correct_step: Dict[str, int] = {p: -1 for p in train_paths_full}
    stats_csv = os.path.join(args.output_dir, "example_statistics.csv")
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "path", "correct"])  # per-step per-sample logging

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        model.train()
        for images, targets, batch_paths in train_loader:
            global_step += 1
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct_mask_batch = (preds == targets).cpu().tolist()

            # Per-step RIGHT/WRONG full-path logging
            right_paths = [p for p, c in zip(batch_paths, correct_mask_batch) if c]
            wrong_paths = [p for p, c in zip(batch_paths, correct_mask_batch) if not c]
            if right_paths:
                print("RIGHT: " + " | ".join(right_paths))
            if wrong_paths:
                print("WRONG: " + " | ".join(wrong_paths))

            # Record first-correct step and append real-time per-sample status
            try:
                with open(stats_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for p, is_corr in zip(batch_paths, correct_mask_batch):
                        if p in first_correct_step and is_corr and first_correct_step[p] == -1:
                            first_correct_step[p] = global_step
                        w.writerow([global_step, p, int(is_corr)])
            except Exception:
                pass

        # Brief epoch summary
        print(f"  epoch_end_loss={float(loss.detach().item()):.4f}")

    # Final check: count remaining not yet correct
    remaining = sum(1 for v in first_correct_step.values() if v == -1)
    print(f"Training done. Examples never correct: {remaining}")

    summary_csv = os.path.join(args.output_dir, "first_correct_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "first_correct_step"])  # -1 means never correct during training
        for p, step in first_correct_step.items():
            w.writerow([p, step])


if __name__ == "__main__":
    main()
