import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from .data import read_imagenet_paths, read_synset_to_index, extract_synset_from_path


class ImageNetEvalDataset(Dataset):
    def __init__(self, image_paths: List[str], synset_to_index: dict, root_dir: str | None, transform) -> None:
        self.image_paths = image_paths
        self.synset_to_index = synset_to_index
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def _resolve(self, idx: int) -> str:
        p = self.image_paths[idx]
        if self.root_dir and not os.path.isabs(p):
            return os.path.join(self.root_dir, p)
        return p

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        path = self._resolve(idx)
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        synset = extract_synset_from_path(path)
        if synset is None or synset not in self.synset_to_index:
            raise RuntimeError(f"Could not determine synset/class for path: {path}")
        y = self.synset_to_index[synset]
        return x, y, idx, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a torchvision ResNet18 on ImageNet validation and save per-image correctness mask in CSV order.")
    parser.add_argument("--examples_csv", type=str, default=os.path.join("training_gradient_evaluator", "imagenet_examples.csv"))
    parser.add_argument("--mapping_txt", type=str, default=os.path.join("image_difficulty_classifier", "imagenet_class_name_mapping.txt"))
    parser.add_argument("--root_dir", type=str, default=None, help="Optional root dir to prepend to relative paths in CSV")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=160)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load torchvision resnet18 with pretrained weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.eval().to(device)

    # Transforms similar to image_difficulty_classifier (using weights' mean/std), but with requested size
    mean = weights.meta["mean"]
    std = weights.meta["std"]
    transform = T.Compose(
        [
            T.Resize(args.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(args.image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    # Read data
    paths = read_imagenet_paths(args.examples_csv)
    if not paths:
        raise RuntimeError(f"No image paths found in {args.examples_csv}")
    synset_to_index = read_synset_to_index(args.mapping_txt)

    ds = ImageNetEvalDataset(paths, synset_to_index, args.root_dir, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    results = np.zeros(len(paths), dtype=bool)
    total = 0
    correct_total = 0

    with torch.no_grad():
        for images, targets, idxs, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct_mask = (preds == targets)

            # Per-batch verbose logging of RIGHT/WRONG full paths
            right_paths = [p for p, c in zip(batch_paths, correct_mask.cpu().tolist()) if c]
            wrong_paths = [p for p, c in zip(batch_paths, correct_mask.cpu().tolist()) if not c]
            if right_paths:
                print("RIGHT: " + " | ".join(right_paths))
            if wrong_paths:
                print("WRONG: " + " | ".join(wrong_paths))

            correct = correct_mask.cpu().numpy().astype(bool)
            idxs_np = idxs.numpy()
            results[idxs_np] = correct
            total += int(targets.numel())
            correct_total += int(correct.sum())

    acc = correct_total / max(total, 1)
    out_dir = os.path.dirname(os.path.abspath(args.examples_csv))
    out_path = os.path.join(out_dir, "resnet_18_160_classification_imagenet_1k.npy")
    np.save(out_path, results)
    print(f"Saved correctness mask (True=correct) to {out_path}. Acc={acc:.4f} ({correct_total}/{total})")


if __name__ == "__main__":
    main()


