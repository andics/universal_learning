import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import timm

try:
    from timm.data import resolve_model_data_config, create_transform
except Exception:  # older timm fallback
    resolve_model_data_config = None  # type: ignore
    create_transform = None  # type: ignore

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
    parser = argparse.ArgumentParser(description="Evaluate a pretrained timm model on ImageNet validation and save per-image correctness mask in CSV order.")
    parser.add_argument("--model_name", type=str, default="resnet_18_160_classification_imagenet_1k")
    parser.add_argument("--examples_csv", type=str, default=os.path.join("training_gradient_evaluator", "imagenet_examples.csv"))
    parser.add_argument("--mapping_txt", type=str, default=os.path.join("image_difficulty_classifier", "imagenet_class_name_mapping.txt"))
    parser.add_argument("--root_dir", type=str, default=None, help="Optional root dir to prepend to relative paths in CSV")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=160, help="Fallback image size if timm data config is unavailable")
    parser.add_argument("--true_means_correct", action="store_true", help="If set, output True for correct (default). If not set, still True=correct; flag kept for symmetry.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model = timm.create_model(args.model_name, pretrained=True, num_classes=1000)
    model.eval().to(device)

    # Resolve transforms from timm when available
    if resolve_model_data_config and create_transform:
        data_cfg = resolve_model_data_config(model)
        transform = create_transform(**data_cfg, is_training=False)
    else:
        import torchvision.transforms as T
        transform = T.Compose(
            [
                T.Resize(args.image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(args.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        for images, targets, idxs, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct = (preds == targets).cpu().numpy().astype(bool)
            idxs_np = idxs.numpy()
            results[idxs_np] = correct
            total += int(targets.numel())
            correct_total += int(correct.sum())

    acc = correct_total / max(total, 1)
    out_dir = os.path.dirname(os.path.abspath(args.examples_csv))
    out_path = os.path.join(out_dir, f"{args.model_name}.npy")
    np.save(out_path, results)
    print(f"Saved correctness mask (True=correct) to {out_path}. Acc={acc:.4f} ({correct_total}/{total})")


if __name__ == "__main__":
    main()


