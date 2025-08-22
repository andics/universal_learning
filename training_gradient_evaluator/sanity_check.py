import argparse
import os, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from training_gradient_evaluator.data import (
	read_imagenet_paths,
	extract_synset_from_path,
)


# Align cwd and sys.path with the project root like train_grad.py
try:
	path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
	if path_main not in sys.path:
		sys.path.append(path_main)
	os.chdir(path_main)
except Exception:
	pass


def _default_hierarchy_json_path() -> str:
	return os.path.join('bars', 'imagenet_synset_hierarchy.json')


def load_imagenet_hierarchy(path: str) -> tuple[dict[str, int], dict[int, str], dict[str, str]]:
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	wnid_to_idx: dict[str, int] = {}
	idx_to_words: dict[int, str] = {}
	wnid_to_words: dict[str, str] = {}
	for wnid, obj in data.items():
		idx = int(obj.get('pytorch_class_id'))
		words = str(obj.get('words', '')).strip()
		wnid_to_idx[wnid] = idx
		wnid_to_words[wnid] = words
		# Do not overwrite if duplicate indices appear; keep first
		if idx not in idx_to_words:
			idx_to_words[idx] = words
	return wnid_to_idx, idx_to_words, wnid_to_words


class ImageNetEvalDataset(Dataset):
	def __init__(self, image_paths: List[str], synset_to_model_idx: dict, root_dir: str | None, transform) -> None:
		self.image_paths = image_paths
		self.synset_to_model_idx = synset_to_model_idx
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self) -> int:
		return len(self.image_paths)

	def _resolve(self, idx: int) -> str:
		p = self.image_paths[idx]
		if self.root_dir and not os.path.isabs(p):
			return os.path.join(self.root_dir, p)
		return p

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
		path = self._resolve(idx)
		image = Image.open(path).convert("RGB")
		x = self.transform(image)
		synset = extract_synset_from_path(path)
		if synset is None or synset not in self.synset_to_model_idx:
			raise RuntimeError(f"Could not map synset to model index for path: {path}")
		y = self.synset_to_model_idx[synset]
		return x, y, idx


def main() -> None:
	parser = argparse.ArgumentParser(description="Sanity check a pretrained timm model on ImageNet validation and compare with bars/imagenet.npy row mask.")
	parser.add_argument("--model_name", type=str, default="resnet18.a3_in1k")
	parser.add_argument("--examples_csv", type=str, default=os.path.join("bars", "imagenet_examples_ammended.csv"))
	parser.add_argument("--hierarchy_json", type=str, default=_default_hierarchy_json_path(), help="Path to bars/imagenet_synset_hierarchy.json")
	parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "imagenet.npy"))
	parser.add_argument("--mask_row_index", type=int, default=1015)
	parser.add_argument("--root_dir", type=str, default=None)
	parser.add_argument("--image_size", type=int, default=160)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--output_dir", type=str, default=os.path.join("training_gradient_evaluator", "outputs"))
	args = parser.parse_args()

	# Resolve output dir per model
	os.makedirs(args.output_dir, exist_ok=True)
	safe_model = args.model_name.replace('/', '_')
	model_out_dir = os.path.join(args.output_dir, safe_model)
	os.makedirs(model_out_dir, exist_ok=True)

	device = torch.device(args.device)

	# Data
	paths = read_imagenet_paths(args.examples_csv)
	if not paths:
		raise RuntimeError(f"No image paths found in {args.examples_csv}")

	# Model and label mappings
	import timm
	model = timm.create_model(args.model_name, pretrained=True)
	# Load mappings from hierarchy
	synset_to_idx, index_to_name, wnid_to_words = load_imagenet_hierarchy(args.hierarchy_json)
	model.eval().to(device)

	# index_to_name already provided by hierarchy mapping

	# Transforms from timm model data_config (eval pipeline)
	data_config = timm.data.resolve_model_data_config(model)
	transform = timm.data.create_transform(**data_config, is_training=False)

	ds = ImageNetEvalDataset(paths, synset_to_idx, args.root_dir, transform)
	loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	# Inference
	correct_mask = np.zeros(len(paths), dtype=bool)
	preds_all = -np.ones(len(paths), dtype=np.int64)
	targets_all = -np.ones(len(paths), dtype=np.int64)
	correct_total = 0
	total = 0
	pbar = tqdm(total=len(paths), desc="Inferring", unit="img")
	with torch.no_grad():
		for images, targets, idxs in loader:
			images = images.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)
			logits = model(images)
			preds = torch.argmax(logits, dim=1)
			idxs_np = idxs.numpy()
			preds_np = preds.cpu().numpy()
			targets_np = targets.cpu().numpy()
			batch_correct = []
			for i, pred_idx in enumerate(preds_np):
				global_idx = int(idxs_np[i])
				path = paths[global_idx]
				if args.root_dir and not os.path.isabs(path):
					path = os.path.join(args.root_dir, path)
				wnid = extract_synset_from_path(path)
				# Numeric correctness based on indices
				tgt_idx = synset_to_idx.get(wnid, -1)
				ok = int(pred_idx) == int(tgt_idx)
				batch_correct.append(ok)
			correct = np.array(batch_correct, dtype=bool)
			correct_mask[idxs_np] = correct
			preds_all[idxs_np] = preds_np
			targets_all[idxs_np] = targets_np
			correct_total += int(correct.sum())
			total += int(len(batch_correct))
			pbar.update(int(len(batch_correct)))
	pbar.close()

	acc = correct_total / max(total, 1)
	print(f"Sanity accuracy: {acc:.4f} ({correct_total}/{total})")

	# Compare with bars/imagenet.npy row
	row_mask = np.load(args.bars_npy)
	if row_mask.ndim != 2 or not (0 <= args.mask_row_index < row_mask.shape[0]):
		raise ValueError(f"Unexpected imagenet.npy shape {row_mask.shape} or bad row {args.mask_row_index}")
	baseline = row_mask[args.mask_row_index].astype(bool)  # True means baseline got it correct
	if baseline.shape[0] != correct_mask.shape[0]:
		print("Warning: length mismatch between CSV paths and npy mask; truncating to min length.")
		min_len = min(baseline.shape[0], correct_mask.shape[0])
		baseline = baseline[:min_len]
		correct_mask = correct_mask[:min_len]
	overlap = float((baseline == correct_mask).mean()) if baseline.size > 0 else float('nan')
	print(f"Percentage overlap with imagenet.npy row {args.mask_row_index}: {overlap*100:.2f}%")

	# Save mask
	out_path = os.path.join(model_out_dir, f"sanity_{safe_model}_correct_mask.npy")
	np.save(out_path, correct_mask)
	print(f"Saved sanity mask to {out_path}")

	# Visualize a random subset of 10 samples with Pred (top) and GT (bottom)
	try:
		rng = np.random.default_rng(42)
		N = len(paths)
		k = min(10, N)
		idx_subset = rng.choice(np.arange(N), size=k, replace=False)
		fig, axes = plt.subplots(2, 5, figsize=(16, 7))
		axes = axes.flatten()
		for ax, i in zip(axes, idx_subset):
			img_path = paths[i]
			if args.root_dir and not os.path.isabs(img_path):
				img_path = os.path.join(args.root_dir, img_path)
			try:
				im = Image.open(img_path).convert("RGB")
				ax.imshow(im)
				ax.axis('off')
				pred_idx = int(preds_all[i])
				wnid = extract_synset_from_path(img_path)
				pred_name = index_to_name.get(pred_idx, str(pred_idx))
				gt_name = wnid_to_words.get(wnid, str(wnid)) if wnid is not None else ""
				ax.set_title(f"Pred: {pred_name}", fontsize=9)
				ax.text(0.5, -0.12, f"GT: {gt_name}", fontsize=9, ha='center', va='top', transform=ax.transAxes)
			except Exception:
				ax.axis('off')
		plt.tight_layout()
		plt.subplots_adjust(bottom=0.12)
		plot_path = os.path.join(model_out_dir, "sanity_random_10.png")
		plt.savefig(plot_path, dpi=150)
		plt.close()
		print(f"Saved random 10-sample visualization to {plot_path}")
	except Exception as e:
		print(f"Warning: failed to render sample visualization: {e}")


if __name__ == "__main__":
	main()
