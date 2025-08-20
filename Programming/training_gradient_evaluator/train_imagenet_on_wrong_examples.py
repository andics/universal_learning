import argparse
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

import timm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def read_imagenet_paths(csv_path: str) -> List[str]:
    # CSV is a single comma-separated line of paths
    with open(csv_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.lstrip("\ufeff").strip()
    if not text:
        return []
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return [p.strip() for p in text.split(",") if p.strip()]


def read_synset_to_index(mapping_path: str) -> Dict[str, int]:
    # mapping file format: "n02119789 1 kit_fox"
    mapping: Dict[str, int] = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            synset = parts[0]
            try:
                idx_1_based = int(parts[1])
            except ValueError:
                continue
            mapping[synset] = idx_1_based - 1  # convert to 0-based
    return mapping


def extract_synset_from_path(path: str) -> Optional[str]:
    # Find token like n######## in the path
    norm = path.replace("\\", "/")
    for part in norm.split("/"):
        if len(part) == 9 and part.startswith("n") and part[1:].isdigit():
            return part
    return None


class ImageNetWrongExamplesDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[str],
        selected_indices: Sequence[int],
        synset_to_index: Dict[str, int],
        transform: Optional[T.Compose] = None,
        root_dir: Optional[str] = None,
    ) -> None:
        self.image_paths = list(image_paths)
        self.indices = list(selected_indices)
        self.synset_to_index = synset_to_index
        self.root_dir = root_dir
        self.transform = transform or T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        idx = self.indices[i]
        path = self.image_paths[idx]
        if self.root_dir and not os.path.isabs(path):
            path = os.path.join(self.root_dir, path)
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        synset = extract_synset_from_path(path)
        if synset is None or synset not in self.synset_to_index:
            raise RuntimeError(f"Could not determine synset/class for path: {path}")
        y = self.synset_to_index[synset]
        return x, y


def build_transforms(image_size: int, is_train: bool) -> T.Compose:
    if is_train:
        aug: List[T.transforms] = []  # type: ignore
        try:
            aug.append(T.RandAugment())
        except Exception:
            try:
                aug.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))
            except Exception:
                pass
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                *aug,
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def filter_existing_indices(paths: Sequence[str], indices: Sequence[int], root_dir: Optional[str]) -> List[int]:
    kept: List[int] = []
    for idx in indices:
        p = paths[idx]
        p_full = os.path.join(root_dir, p) if root_dir and not os.path.isabs(p) else p
        if os.path.exists(p_full):
            kept.append(idx)
    return kept


def create_dataloaders(
    image_paths: List[str],
    wrong_indices: List[int],
    synset_to_index: Dict[str, int],
    root_dir: Optional[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    # simple random split: 90% train, 10% val
    rng = np.random.default_rng(12345)
    wrong_indices = list(wrong_indices)
    rng.shuffle(wrong_indices)
    if len(wrong_indices) < 10:
        split = len(wrong_indices)
    else:
        split = int(0.9 * len(wrong_indices))
    train_idx = wrong_indices[:split]
    val_idx = wrong_indices[split:]

    train_ds = ImageNetWrongExamplesDataset(
        image_paths=image_paths,
        selected_indices=train_idx,
        synset_to_index=synset_to_index,
        transform=build_transforms(image_size, is_train=True),
        root_dir=root_dir,
    )
    val_ds = ImageNetWrongExamplesDataset(
        image_paths=image_paths,
        selected_indices=val_idx,
        synset_to_index=synset_to_index,
        transform=build_transforms(image_size, is_train=False),
        root_dir=root_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res: List[torch.Tensor] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, scaler: Optional[torch.cuda.amp.GradScaler], log_interval: int = 50) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0
    start = time.time()
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        running_loss += loss.item()
        running_acc += acc1.item()
        num_batches += 1

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - start
            print(f"  [train] step {step + 1}/{len(loader)} loss={running_loss / num_batches:.4f} acc1={running_acc / num_batches:.2f}% ({elapsed:.1f}s)", flush=True)

    return running_loss / max(1, num_batches), running_acc / max(1, num_batches)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        running_loss += loss.item()
        running_acc += acc1.item()
        num_batches += 1
    return running_loss / max(1, num_batches), running_acc / max(1, num_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a timm ImageNet model only on images the selected model got wrong.")
    parser.add_argument("--model_name", type=str, default="mobilenetv3_small_050_224_lamb_imagenet_1k", help="timm model name to train")
    parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "imagenet.npy"), help="Path to bars/imagenet.npy mask array")
    parser.add_argument("--examples_csv", type=str, default=os.path.join("image_difficulty_classifier", "imagenet_examples.csv"), help="Path to CSV with ImageNet image paths")
    parser.add_argument("--mapping_txt", type=str, default=os.path.join("image_difficulty_classifier", "imagenet_class_name_mapping.txt"), help="Path to synset->index mapping text file")
    parser.add_argument("--root_dir", type=str, default=None, help="Optional root directory to prepend to relative image paths")
    parser.add_argument("--mask_row_index", type=int, default=1023, help="Row index in imagenet.npy to use (worst model is 1023 by default; reverse-ordered)")
    parser.add_argument("--mask_true_means_wrong", action="store_true", help="If set, True in npy mask means the image was wrong. If not set, False means wrong.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=os.path.join("training_gradient_evaluator", "outputs"))
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 to only save best)")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading paths and mappings...", flush=True)
    image_paths = read_imagenet_paths(args.examples_csv)
    synset_to_index = read_synset_to_index(args.mapping_txt)

    print(f"Loading mask from {args.bars_npy} ...", flush=True)
    mask_array = np.load(args.bars_npy)
    if mask_array.ndim == 1:
        # shape: [num_examples]
        selected_mask = mask_array.astype(bool)
    elif mask_array.ndim == 2:
        # shape: [num_models, num_examples]
        if args.mask_row_index < 0:
            row = mask_array.shape[0] + args.mask_row_index
        else:
            row = args.mask_row_index
        if row < 0 or row >= mask_array.shape[0]:
            raise IndexError(f"mask_row_index {args.mask_row_index} is out of bounds for array with shape {mask_array.shape}")
        selected_mask = mask_array[row].astype(bool)
    else:
        raise ValueError(f"Unexpected imagenet.npy shape: {mask_array.shape}")

    # Interpret meaning: by default, use --mask_true_means_wrong to specify True==wrong
    if args.mask_true_means_wrong:
        wrong_mask = selected_mask
    else:
        wrong_mask = ~selected_mask

    wrong_indices = np.nonzero(wrong_mask)[0].tolist()
    print(f"Selected {len(wrong_indices)} wrong examples before file filtering.")

    wrong_indices = filter_existing_indices(image_paths, wrong_indices, args.root_dir)
    if len(wrong_indices) == 0:
        raise RuntimeError("No existing image files found among selected wrong examples. Check --root_dir and paths.")
    print(f"{len(wrong_indices)} wrong examples after filtering for existing files.")

    train_loader, val_loader = create_dataloaders(
        image_paths=image_paths,
        wrong_indices=wrong_indices,
        synset_to_index=synset_to_index,
        root_dir=args.root_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = 1000
    print(f"Creating model {args.model_name} ...", flush=True)
    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)

    device = torch.device(args.device)
    model = model.to(device)
    if torch.cuda.device_count() > 1 and args.device.startswith("cuda"):
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = None if args.no_amp or not (device.type == "cuda") else torch.cuda.amp.GradScaler()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}_{args.model_name}")
    os.makedirs(run_dir, exist_ok=True)

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(run_dir, "best.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"  train: loss={train_loss:.4f} acc1={train_acc:.2f}% | val: loss={val_loss:.4f} acc1={val_acc:.2f}%")

        # Save checkpoints
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "val_acc1": val_acc,
            "train_acc1": train_acc,
            "args": vars(args),
        }
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(state, best_ckpt_path)
            print(f"  Saved new best checkpoint to {best_ckpt_path}")
        if args.save_every and (epoch % args.save_every == 0):
            ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pth")
            torch.save(state, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

    print(f"Training complete. Best val acc1: {best_val_acc:.2f}%. Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()


