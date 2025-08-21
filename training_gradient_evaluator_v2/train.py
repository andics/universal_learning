import argparse
import os, sys
from pathlib import Path
from typing import List
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training_gradient_evaluator_v2.data import ImageNetWrongExamplesDataset, read_imagenet_paths, read_synset_to_index, build_transforms
from training_gradient_evaluator_v2.engine import TrainConfig, Trainer
from training_gradient_evaluator_v2.utils import create_logger


def filter_existing_indices(paths: List[str], indices: List[int], root_dir: str | None) -> List[int]:
	kept: List[int] = []
	for idx in indices:
		p = paths[idx]
		full = os.path.join(root_dir, p) if root_dir and not os.path.isabs(p) else p
		if os.path.exists(full):
			kept.append(idx)
	return kept


def main() -> None:
	parser = argparse.ArgumentParser(description="V2 OOP trainer using torchvision transforms")
	parser.add_argument("--model_name", type=str, default="mobilenetv3_small_050.lamb_in1k")
	parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "imagenet.npy"))
	parser.add_argument("--examples_csv", type=str, default=os.path.join("bars", "imagenet_examples_ammended.csv"))
	parser.add_argument("--mapping_txt", type=str, default=os.path.join("image_difficulty_classifier", "imagenet_class_name_mapping.txt"))
	parser.add_argument("--root_dir", type=str, default=None)
	parser.add_argument("--mask_row_index", type=int, default=1022)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--lr", type=float, default=5e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-2)
	parser.add_argument("--image_size", type=int, default=224)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--output_dir", type=str, default=os.path.join("training_gradient_evaluator_v2", "outputs"))
	parser.add_argument("--no_amp", action="store_true")
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	logger = create_logger(args.output_dir)

	paths = read_imagenet_paths(args.examples_csv)
	synset_to_index = read_synset_to_index(args.mapping_txt)

	mask = np.load(args.bars_npy)
	row = args.mask_row_index
	wrong = ~mask[row].astype(bool)
	wrong_indices = np.nonzero(wrong)[0].tolist()
	wrong_indices = filter_existing_indices(paths, wrong_indices, args.root_dir)
	if not wrong_indices:
		raise RuntimeError("No existing images among wrong examples")

	# Split for OOP trainer (80/10/10)
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

	# Build model via timm (weights) but training side uses TV transforms
	import timm
	model = timm.create_model(args.model_name, pretrained=True, num_classes=1000)
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


