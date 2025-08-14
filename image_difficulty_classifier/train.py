import argparse
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
            "No samples found from CSV. The file may be empty or the format is unexpected."
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
    parser.add_argument("--csv", default="/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/image_difficulty_classifier/imagenet_examples.csv",
                        required=False, help="Path to imagenet_examples.csv")
    parser.add_argument("--output-dir", default="/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/image_difficulty_classifier/output",
                        required=False, help="Directory to write logs/checkpoints")
    parser.add_argument("--model-name", default="clip_mlp", choices=list_models())
    parser.add_argument("--clip-backbone", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b82k")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every-fraction", type=float, default=0.2)
    parser.add_argument("--no-resume", action="store_true", help="Do not resume even if a checkpoint exists")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--root-dir", type=str, default=None, help="Optional root dir to prefix image paths from CSV")
    args = parser.parse_args()

    # Automatically append model name to output directory for better organization
    base_output_dir = args.output_dir
    if args.model_name == "clip_linear" or args.model_name == "clip_mlp":
        # For CLIP models, include both model type and backbone info
        model_suffix = f"_{args.model_name.upper()}_{args.clip_backbone.replace('-', '_')}_{args.clip_pretrained.replace('-', '_')}"
        args.output_dir = base_output_dir + model_suffix
    else:
        # For other models, just append the model name
        args.output_dir = base_output_dir + f"_{args.model_name.upper()}"

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(name="train")
    set_global_seed(args.seed)
    
    # Log the output directory being used
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model_name}")
    if args.model_name in ["clip_linear", "clip_mlp"]:
        logger.info(f"CLIP backbone: {args.clip_backbone}")
        logger.info(f"CLIP pretrained: {args.clip_pretrained}")

    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path=args.csv,
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

    # Improved cosine schedule with warmup (10% of total steps) starting from higher LR
    total_steps = args.epochs * math.ceil(len(train_loader.dataset) / args.batch_size)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            # Warmup from 0 to peak LR (which is 3x the base LR)
            return 3.0 * float(step) / float(max(1, warmup_steps))
        # Cosine decay from 3x base LR to 0.1x base LR
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_lr_ratio = 0.1
        return min_lr_ratio + (3.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

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