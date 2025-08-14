import argparse
import csv
import json
import math
import os
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from image_difficulty_classifier.data import ImageNetDifficultyBinDataset, default_transforms
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
):
    # Determine total items from CSV using the dataset CSV reader
    all_paths = ImageNetDifficultyBinDataset._read_csv(csv_path)
    num_items = len(all_paths)
    if num_items == 0:
        raise ValueError(
            "No samples found from CSV. The file may be empty or all paths were filtered/mismatched. "
            "If you used --path-prefix, ensure it matches the beginning of paths in the CSV."
        )

    train_idx, val_idx, test_idx = split_indices(num_items, seed)
    transform = default_transforms(image_size)

    train_ds = ImageNetDifficultyBinDataset(csv_path, train_idx, transform, root_dir=root_dir)
    val_ds = ImageNetDifficultyBinDataset(csv_path, val_idx, transform, root_dir=root_dir)
    test_ds = ImageNetDifficultyBinDataset(csv_path, test_idx, transform, root_dir=root_dir)

    # Avoid RandomSampler crash on empty training set (shouldn't happen now, but guard anyway)
    train_shuffle = True if len(train_ds) > 0 else False
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
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
    parser.add_argument("--csv", default="/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/imagenet_examples.csv",
                        required=False, help="Path to imagenet_examples.csv")
    parser.add_argument("--output-dir", default="/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/image_difficulty_classifier/output",
                        required=False, help="Directory to write logs/checkpoints")
    parser.add_argument("--model-name", default="clip_linear", choices=list_models())
    parser.add_argument("--clip-backbone", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every-fraction", type=float, default=0.2)
    parser.add_argument("--no-resume", action="store_true", help="Do not resume even if a checkpoint exists")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--root-dir", type=str, default=None, help="Optional root dir to prefix image paths from CSV")
    parser.add_argument(
        "--path-prefix",
        type=str,
        default="/home/projects/bagon/shared/imagenet512/val",
        help=(
            "Optional replacement for the initial prefix '/om2/user/cheungb/datasets/imagenet_validation/val'"
            " in the CSV paths. If set, a copy of the CSV with the prefix replaced will be written to the"
            " package folder and used for training."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(name="train")
    set_global_seed(args.seed)

    # If a path prefix is provided, create a local copy of the CSV with the prefix replaced
    csv_path_to_use = args.csv
    if args.path_prefix:
        old_prefix = "/om2/user/cheungb/datasets/imagenet_validation/val"

        def _replace_prefix(p: str) -> str:
            # Replace all occurrences of the old prefix within the string (handles comma-concatenated paths)
            return p.replace(old_prefix, args.path_prefix)

        module_dir = os.path.dirname(__file__)
        rewritten_csv_path = os.path.join(module_dir, "imagenet_examples.csv")

        # Read input CSV, detect header, and write output while replacing paths
        with open(args.csv, "r", newline="", encoding="utf-8") as fin:
            first_line = fin.readline()
            fin.seek(0)
            if "path" in [h.strip() for h in first_line.strip().split(",")]:
                reader = csv.DictReader(fin)
                fieldnames = reader.fieldnames or ["path"]
                with open(rewritten_csv_path, "w", newline="", encoding="utf-8") as fout:
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in reader:
                        if "path" in row and row["path"]:
                            row["path"] = _replace_prefix(row["path"])  # type: ignore[index]
                        writer.writerow(row)
            else:
                reader2 = csv.reader(fin)
                with open(rewritten_csv_path, "w", newline="", encoding="utf-8") as fout2:
                    writer2 = csv.writer(fout2)
                    for row in reader2:
                        if not row:
                            writer2.writerow(row)
                            continue
                        row0 = row[0]
                        row[0] = _replace_prefix(row0)
                        writer2.writerow(row)

        csv_path_to_use = rewritten_csv_path

    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path=csv_path_to_use,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        root_dir=args.root_dir,
    )

    model = get_model(
        args.model_name,
        num_classes=5,
        clip_backbone=args.clip_backbone,
        clip_pretrained=args.clip_pretrained,
        unfreeze_backbone=args.unfreeze_backbone,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine schedule with warmup (5% of total steps)
    total_steps = args.epochs * math.ceil(len(train_loader.dataset) / args.batch_size)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

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