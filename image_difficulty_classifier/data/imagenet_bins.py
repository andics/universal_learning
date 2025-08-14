from dataclasses import dataclass
import os
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T


@dataclass
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]


def indices_to_bins(index: int) -> int:
    # 0..9999 -> 0; 10000..19999 -> 1; ...; 40000..49999 -> 4
    return min(index // 10000, 4)


def default_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )


class ImageNetDifficultyBinDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        indices: List[int],
        transform: Optional[Callable] = None,
        root_dir: Optional[str] = None,
    ) -> None:
        self.image_paths: List[str] = self._read_csv(csv_path)
        self.indices: List[int] = indices
        self.transform = transform or default_transforms()
        self.root_dir = root_dir

    @staticmethod
    def _read_csv(csv_path: str) -> List[str]:
        with open(csv_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.lstrip("\ufeff").strip()
        if not text:
            return []
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        return [p.strip() for p in text.split(",") if p.strip()]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        idx = self.indices[i]
        path = self.image_paths[idx]
        if self.root_dir and not os.path.isabs(path):
            path = os.path.join(self.root_dir, path)
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        y = indices_to_bins(idx)
        return x, y


