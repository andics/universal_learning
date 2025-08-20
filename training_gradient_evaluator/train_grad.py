import argparse
import os, sys
from typing import List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

from training_gradient_evaluator.data import ImageNetWrongExamplesDataset, build_transforms, read_imagenet_paths, read_synset_to_index
from training_gradient_evaluator.engine import TrainConfig, Trainer
from training_gradient_evaluator.models import create_default_model
from training_gradient_evaluator.utils import create_logger


def filter_existing_indices(paths: List[str], indices: List[int], root_dir: str | None) -> List[int]:
	kept: List[int] = []
	for idx in indices:
		p = paths[idx]
		full = os.path.join(root_dir, p) if root_dir and not os.path.isabs(p) else p
		if os.path.exists(full):
			kept.append(idx)
	return kept


def main() -> None:
	parser = argparse.ArgumentParser(description="Modular trainer on wrong ImageNet examples from bars/imagenet.npy")
	parser.add_argument("--model_name", type=str, default="mobilenetv3_small_050_224_lamb_imagenet_1k")
	parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "imagenet.npy"))
	parser.add_argument("--examples_csv", type=str, default=os.path.join("image_difficulty_classifier", "imagenet_examples.csv"))
	parser.add_argument("--mapping_txt", type=str, default=os.path.join("image_difficulty_classifier", "imagenet_class_name_mapping.txt"))
	parser.add_argument("--root_dir", type=str, default=None)
	parser.add_argument("--mask_row_index", type=int, default=1022, help="Worst model row is 1022 (reverse-ordered)")
	parser.add_argument("--mask_true_means_wrong", action="store_true")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--lr", type=float, default=5e-4)
	parser.add_argument("--weight_decay", type=float, default=5e-2)
	parser.add_argument("--image_size", type=int, default=224)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--output_dir", type=str, default=os.path.join("training_gradient_evaluator", "outputs"))
	parser.add_argument("--no_amp", action="store_true")
	parser.add_argument("--gradient_clip", type=float, default=None)
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	logger = create_logger(args.output_dir)

	paths = read_imagenet_paths(args.examples_csv)
	synset_to_index = read_synset_to_index(args.mapping_txt)

	mask = np.load(args.bars_npy)
	if mask.ndim == 2:
		row = args.mask_row_index if args.mask_row_index >= 0 else mask.shape[0] + args.mask_row_index
		selected = mask[row].astype(bool)
	else:
		selected = mask.astype(bool)
	wrong = selected if args.mask_true_means_wrong else ~selected
	wrong_indices = np.nonzero(wrong)[0].tolist()
	wrong_indices = filter_existing_indices(paths, wrong_indices, args.root_dir)
	if not wrong_indices:
		raise RuntimeError("No existing images found for selected wrong examples")

	# Split train/val/test: 80/10/10
	rng = np.random.default_rng(123)
	rng.shuffle(wrong_indices)
	n = len(wrong_indices)
	tr_n = int(0.8 * n)
	va_n = int(0.1 * n)
	train_idx = wrong_indices[:tr_n]
	val_idx = wrong_indices[tr_n : tr_n + va_n]
	test_idx = wrong_indices[tr_n + va_n :]

	train_ds = ImageNetWrongExamplesDataset(paths, train_idx, synset_to_index, transform=build_transforms(args.image_size, True), root_dir=args.root_dir)
	val_ds = ImageNetWrongExamplesDataset(paths, val_idx, synset_to_index, transform=build_transforms(args.image_size, False), root_dir=args.root_dir)
	test_ds = ImageNetWrongExamplesDataset(paths, test_idx, synset_to_index, transform=build_transforms(args.image_size, False), root_dir=args.root_dir)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
	val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=max(0, args.num_workers // 2), pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=max(0, args.num_workers // 2), pin_memory=True)

	model = create_default_model(args.model_name, num_classes=1000)
	if torch.cuda.device_count() > 1 and args.device.startswith("cuda"):
		model = torch.nn.DataParallel(model)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = None
	scaler = None if args.no_amp or args.device.startswith("cpu") else torch.cuda.amp.GradScaler()

	config = TrainConfig(
		output_dir=args.output_dir,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		weight_decay=args.weight_decay,
		num_workers=args.num_workers,
		device=args.device,
		gradient_clip_norm=args.gradient_clip,
		log_image_paths=True,
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
		train_image_paths=[paths[i] for i in train_idx],
	)

	trainer.fit()


if __name__ == "__main__":
	main()


