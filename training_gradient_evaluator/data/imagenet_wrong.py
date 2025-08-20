import os
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def read_imagenet_paths(csv_path: str) -> List[str]:
	with open(csv_path, "r", encoding="utf-8") as f:
		text = f.read()
	text = text.lstrip("\ufeff").strip()
	if not text:
		return []
	if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
		text = text[1:-1]
	return [p.strip() for p in text.split(",") if p.strip()]


def read_synset_to_index(mapping_path: str) -> Dict[str, int]:
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
			mapping[synset] = idx_1_based - 1
	return mapping


def extract_synset_from_path(path: str) -> Optional[str]:
	norm = path.replace("\\", "/")
	for part in norm.split("/"):
		if len(part) == 9 and part.startswith("n") and part[1:].isdigit():
			return part
	return None


def gbuild_transforms(image_size: int, is_train: bool) -> T.Compose:
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
		self.transform = transform or build_transforms(224, is_train=True)

	def __len__(self) -> int:
		return len(self.indices)

	def _resolve_path(self, idx: int) -> str:
		p = self.image_paths[idx]
		if self.root_dir and not os.path.isabs(p):
			return os.path.join(self.root_dir, p)
		return p

	def get_item_with_path(self, i: int) -> Tuple[torch.Tensor, int, str]:
		idx = self.indices[i]
		path = self._resolve_path(idx)
		image = Image.open(path).convert("RGB")
		x = self.transform(image)
		synset = extract_synset_from_path(path)
		if synset is None or synset not in self.synset_to_index:
			raise RuntimeError(f"Could not determine synset/class for path: {path}")
		y = self.synset_to_index[synset]
		return x, y, path

	def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
		return self.get_item_with_path(i)


