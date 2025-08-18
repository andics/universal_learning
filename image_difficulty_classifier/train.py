import argparse
import json
import math
import os
import time
import sys
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
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from image_difficulty_classifier.data import ImageNetDifficultyBinDataset, default_transforms, indices_to_bins
from image_difficulty_classifier.engine import Trainer, TrainConfig
from image_difficulty_classifier.models import get_model, list_models
from image_difficulty_classifier.utils.logging import setup_logging
from image_difficulty_classifier.utils.seed import set_global_seed


def split_indices(num_items: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    if num_items <= 0:
        return [], [], []
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(num_items, generator=g).tolist()
    n_train = math.floor(num_items * 0.85)
    n_val = math.floor(num_items * 0.05)
    n_test = num_items - n_train - n_val
    if n_train == 0 and num_items > 0:
        if n_test > 0:
            n_train, n_test = 1, n_test - 1
        elif n_val > 0:
            n_train, n_val = 1, n_val - 1
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def build_dataloaders(
    csv_path: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    image_size: int,
    root_dir: Optional[str] = None,
    num_bins: int = 5,
    min_images_common: Optional[int] = None,
):
    # Determine total items from CSV using the dataset CSV reader
    all_paths = ImageNetDifficultyBinDataset._read_csv(csv_path)
    num_items = len(all_paths)
    if num_items == 0:
        raise ValueError(
            "No samples found from CSV. The file may be empty or the format is unexpected."
        )

    # Helper: extract ImageNet class id from path
    import re

    def extract_class_id(path: str) -> str:
        m = re.search(r"/(n\d{8})/", path)
        return m.group(1) if m else "unknown"

    # Helper: bin assignment by index
    half_point = num_items // 2

    def assign_bin(index: int) -> int:
        if num_bins == 2:
            return 0 if index < half_point else 1
        # 5-bin default via shared helper
        return indices_to_bins(index)

    # Optionally filter to classes with minimum images per bin >= threshold
    # Build class -> bin counts
    if min_images_common is not None and min_images_common > 0:
        class_to_counts: Dict[str, List[int]] = {}
        for idx, path in enumerate(all_paths):
            cls = extract_class_id(path)
            if cls not in class_to_counts:
                class_to_counts[cls] = [0] * num_bins
            class_to_counts[cls][assign_bin(idx)] += 1

        # Keep only classes present with at least min_images_common in every bin
        kept_classes = {c for c, counts in class_to_counts.items() if len(counts) == num_bins and min(counts) >= min_images_common}

        # If nothing left, fall back to no filtering
        if len(kept_classes) == 0:
            kept_classes = None  # no filtering
        else:
            # Build filtered index list
            filtered_indices = [i for i, p in enumerate(all_paths) if extract_class_id(p) in kept_classes]
            # Overwrite num_items for splitting scope
            num_items_filtered = len(filtered_indices)
            if num_items_filtered == 0:
                train_idx, val_idx, test_idx = split_indices(num_items, seed)
            else:
                # Shuffle filtered indices and split
                g = torch.Generator()
                g.manual_seed(seed)
                perm = torch.randperm(num_items_filtered, generator=g).tolist()
                filtered_indices = [filtered_indices[i] for i in perm]
                n_train = math.floor(num_items_filtered * 0.85)
                n_val = math.floor(num_items_filtered * 0.05)
                n_test = num_items_filtered - n_train - n_val
                train_idx = filtered_indices[:n_train]
                val_idx = filtered_indices[n_train : n_train + n_val]
                test_idx = filtered_indices[n_train + n_val :]
        
        if 'train_idx' not in locals():
            train_idx, val_idx, test_idx = split_indices(num_items, seed)
    else:
        train_idx, val_idx, test_idx = split_indices(num_items, seed)
    # Use the same deterministic transforms for all splits (no augmentation)
    transform_eval = default_transforms(image_size)

    train_ds = ImageNetDifficultyBinDataset(csv_path, train_idx, transform_eval, root_dir=root_dir)
    val_ds = ImageNetDifficultyBinDataset(csv_path, val_idx, transform_eval, root_dir=root_dir)
    test_ds = ImageNetDifficultyBinDataset(csv_path, test_idx, transform_eval, root_dir=root_dir)

    # If using 2-bin mode, wrap datasets to relabel based on equal halves
    if num_bins == 2:
        class TwoBinLabelWrapper(torch.utils.data.Dataset):
            def __init__(self, base_ds: ImageNetDifficultyBinDataset, total_items: int):
                self.base = base_ds
                self.total = total_items
                self.half = total_items // 2

            def __len__(self) -> int:
                return len(self.base)

            def __getitem__(self, i: int):
                x, _ = self.base[i]
                original_index = self.base.indices[i]
                y2 = 0 if original_index < self.half else 1
                return x, y2

        train_ds = TwoBinLabelWrapper(train_ds, num_items)
        val_ds = TwoBinLabelWrapper(val_ds, num_items)
        test_ds = TwoBinLabelWrapper(test_ds, num_items)

    # Avoid RandomSampler crash on empty training set (shouldn't happen now, but guard anyway)
    # Build a balanced sampler across bins within the selected classes
    if len(train_ds) > 0:
        # collect labels for train set without loading images
        labels = []
        if num_bins == 2:
            for idx in train_idx:
                labels.append(0 if idx < half_point else 1)
        else:
            for idx in train_idx:
                labels.append(indices_to_bins(idx))
        bin_counts = [max(1, labels.count(b)) for b in range(num_bins)]
        weights = [1.0 / bin_counts[y] for y in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
    else:
        sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    eval_loader_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, **eval_loader_kwargs)
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train image difficulty classifier (5 classes)")
    parser.add_argument("--csv", default="/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/image_difficulty_classifier/imagenet_examples.csv",
                        required=False, help="Path to imagenet_examples.csv")
    parser.add_argument("--output-dir", default="/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/image_difficulty_classifier/output",
                        required=False, help="Directory to write logs/checkpoints")
    # Default to frozen CLIP linear head
    parser.add_argument("--model-name", default="clip_linear", choices=list_models())
    parser.add_argument("--clip-backbone", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--unfreeze-backbone", action="store_true", default=False)
    parser.add_argument("--freeze-backbone", dest="unfreeze_backbone", action="store_false", help="Keep backbone frozen (opposite of --unfreeze-backbone)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every-fraction", type=float, default=0.2)
    parser.add_argument("--no-resume", action="store_true", help="Do not resume even if a checkpoint exists")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--root-dir", type=str, default=None, help="Optional root dir to prefix image paths from CSV")
    # New options
    parser.add_argument("--num-bins", type=int, default=2, choices=[2,5], help="Number of difficulty bins to classify (2 = equal halves by index, 5 = original bins)")
    parser.add_argument("--minimum-images-common", type=int, default=19, help="Keep only classes whose per-bin minimum image count is >= this number")
    args = parser.parse_args()

    # Automatically append model name to output directory for better organization
    base_output_dir = args.output_dir
    if args.model_name == "clip_linear" or args.model_name == "clip_mlp":
        # For CLIP models, include both model type and backbone info
        model_suffix = f"_{args.model_name.upper()}_{args.clip_backbone.replace('-', '_')}_{args.clip_pretrained.replace('-', '_')}"
        args.output_dir = base_output_dir + model_suffix
    else:
        # For other models, append model name and fine-tuning status
        tuning_status = "FINETUNE" if args.unfreeze_backbone else "FROZEN"
        args.output_dir = base_output_dir + f"_{args.model_name.upper()}_{tuning_status}"

    # Append key hyperparameters to make runs self-describing
    hyper_suffix = (
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}"
        f"_lr{args.lr}"
        f"_wd{args.weight_decay}"
        f"_nb{args.num_bins}"
        f"_min{args.minimum_images_common}"
        f"_is{args.image_size}"
        f"_nw{args.num_workers}"
        f"_seed{args.seed}"
    )
    args.output_dir = args.output_dir + hyper_suffix

    # Ensure unique output directory by appending _1, _2, ... if needed
    def _make_unique_dir(path: str) -> str:
        if not os.path.exists(path):
            return path
        base = path
        suffix = 1
        while True:
            candidate = f"{base}_{suffix}"
            if not os.path.exists(candidate):
                return candidate
            suffix += 1

    args.output_dir = _make_unique_dir(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    # Save logs in the model directory with absolute epoch timestamp
    logger = setup_logging(name="train", log_dir=args.output_dir)
    set_global_seed(args.seed)

    # Detailed run summary at the very beginning of the log
    run_summary = {
        "timestamp_epoch": int(time.time()),
        "output_dir": args.output_dir,
        "csv": args.csv,
        "root_dir": args.root_dir,
        "data": {
            "image_size": args.image_size,
            "num_workers": args.num_workers,
            "num_bins": args.num_bins,
            "minimum_images_common": args.minimum_images_common,
        },
        "model": {
            "model_name": args.model_name,
            "clip_backbone": args.clip_backbone,
            "clip_pretrained": args.clip_pretrained,
            "unfreeze_backbone": bool(args.unfreeze_backbone),
            "num_classes": args.num_bins,
        },
        "train": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "checkpoint_every_fraction": args.checkpoint_every_fraction,
            "optimizer": "AdamW",
            "scheduler": "LambdaLR(cosine warmup 10%)",
            "amp": bool(torch.cuda.is_available()),
            "seed": args.seed,
        },
    }
    logger.info("RUN CONFIG:\n" + json.dumps(run_summary, indent=2))

    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path=args.csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        root_dir=args.root_dir,
        num_bins=args.num_bins,
        min_images_common=args.minimum_images_common,
    )

    # Build model with appropriate arguments based on model type
    if args.model_name in ["clip_linear", "clip_mlp"]:
        model = get_model(
            args.model_name,
            num_classes=5,
            clip_backbone=args.clip_backbone,
            clip_pretrained=args.clip_pretrained,
            unfreeze_backbone=args.unfreeze_backbone,
        )
    else:
        # For torchvision models (ResNet, ViT), only pass unfreeze_backbone
        model = get_model(
            args.model_name,
            num_classes=5,
            unfreeze_backbone=args.unfreeze_backbone,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )

    # Improved cosine schedule with warmup (10% of total steps) starting from higher LR
    total_steps = args.epochs * math.ceil(len(train_loader.dataset) / args.batch_size)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            # Gentler warmup from 0 to peak LR (which is 2x the base LR for fine-tuning)
            return 2.0 * float(step) / float(max(1, warmup_steps))
        # Cosine decay from 2x base LR to 0.05x base LR (lower minimum for fine-tuning)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_lr_ratio = 0.05
        return min_lr_ratio + (2.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Use new torch.amp API to avoid deprecation warnings
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    config = TrainConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=device,
        checkpoint_every_fraction=args.checkpoint_every_fraction,
        gradient_clip_norm=args.grad_clip,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        scaler=scaler,
        logger=logger, 
    )

    trainer.maybe_resume(resume=not args.no_resume)
    test_metrics = trainer.fit()

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()