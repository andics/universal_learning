import argparse
import csv
import os, sys
from typing import Dict, List
from pathlib import Path
import time
import json
import glob

# Ensure working directory and sys.path point to the Programming root so package imports resolve
try:
	path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
	if path_main not in sys.path:
		sys.path.append(path_main)
	os.chdir(path_main)
except Exception:
	pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoModel

from training_gradient_evaluator.data import ImageNetWrongExamplesDataset, read_imagenet_paths, read_synset_to_index, build_transforms


def filter_existing_indices(paths: List[str], indices: List[int], root_dir: str | None) -> List[int]:
	kept: List[int] = []
	for idx in indices:
		p = paths[idx]
		full = os.path.join(root_dir, p) if root_dir and not os.path.isabs(p) else p
		if os.path.exists(full):
			kept.append(idx)
	return kept


def main() -> None:
	parser = argparse.ArgumentParser(description="V2: Train model on only the images it originally got wrong; torchvision transforms.")
	parser.add_argument("--model_name", type=str, default="timm/resnet18.a3_in1k")
	parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "imagenet.npy"))
	parser.add_argument("--examples_csv", type=str, default=os.path.join("bars", "imagenet_examples_ammended.csv"))
	parser.add_argument("--mapping_txt", type=str, default=os.path.join("image_difficulty_classifier", "imagenet_class_name_mapping.txt"))
	parser.add_argument("--root_dir", type=str, default=None)
	parser.add_argument("--mask_row_index", type=int, default=1015, help="Row for mobilenetv3_small_050.lamb_in1k in imagenet.npy")
	parser.add_argument("--epochs", type=int, default=80)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--lr", type=float, default=5e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-2)
	parser.add_argument("--image_size", type=int, default=160)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--output_dir", type=str, default=os.path.join("training_gradient_evaluator", "outputs"))
	parser.add_argument("--no_amp", action="store_true")
	parser.add_argument("--streak_epochs", type=int, default=10, help="Consecutive epochs required for an example to be considered first-correct")
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	safe_model_name = args.model_name.replace('/', '_')
	model_out_dir = os.path.join(args.output_dir, safe_model_name)
	os.makedirs(model_out_dir, exist_ok=True)
	ckpt_dir = os.path.join(model_out_dir, "checkpoints")
	os.makedirs(ckpt_dir, exist_ok=True)
	device = torch.device(args.device)

	# Paths and labels
	paths = read_imagenet_paths(args.examples_csv)
	if not paths:
		raise RuntimeError(f"No image paths found in {args.examples_csv}")
	synset_to_index = read_synset_to_index(args.mapping_txt)

	# Load original correctness mask and select wrong examples for the specified model row
	mask = np.load(args.bars_npy)
	if mask.ndim != 2 or args.mask_row_index < 0 or args.mask_row_index >= mask.shape[0]:
		raise ValueError(f"Unexpected mask shape {mask.shape} or bad row {args.mask_row_index}")
	correct_mask = mask[args.mask_row_index].astype(bool)  # True means originally correct
	wrong_mask = ~correct_mask
	wrong_indices = np.nonzero(wrong_mask)[0].tolist()
	wrong_indices = filter_existing_indices(paths, wrong_indices, args.root_dir)
	if not wrong_indices:
		raise RuntimeError("No existing images among wrong examples.")
	print(f"Training on {len(wrong_indices)} originally-wrong examples.")

	# Build model (timm) and torchvision transforms (V2 requirement)
	import timm
	model = AutoModel.from_pretrained(args.model_name)
	model = model.to(device)
	if torch.cuda.device_count() > 1 and device.type == "cuda":
		model = nn.DataParallel(model)

	train_tfms = build_transforms(args.image_size, is_train=True)
	# eval_tfms = build_transforms(args.image_size, is_train=False)  # not used in this minimal V2 loop

	# Dataset and loaders
	train_ds = ImageNetWrongExamplesDataset(paths, wrong_indices, synset_to_index, transform=train_tfms, root_dir=args.root_dir)
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

	# Optimizer and AMP
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scaler = None if args.no_amp or device.type != "cuda" else torch.cuda.amp.GradScaler()

	# Stats logging
	def resolve_full(p: str) -> str:
		return os.path.join(args.root_dir, p) if args.root_dir and not os.path.isabs(p) else p

	train_paths_full = [resolve_full(paths[i]) for i in wrong_indices]
	# Track per-example consecutive epoch correctness and the step when it first reached the streak threshold
	consecutive_epoch_correct: Dict[str, int] = {p: 0 for p in train_paths_full}
	tenth_epoch_streak_step: Dict[str, int] = {p: -1 for p in train_paths_full}
	stats_csv = os.path.join(model_out_dir, "example_statistics.csv")
	with open(stats_csv, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["step", "path", "correct"])  # per-step per-sample logging

	# Save hyperparameters
	hparams_path = os.path.join(model_out_dir, "hparams.json")
	try:
		with open(hparams_path, "w", encoding="utf-8") as jf:
			json.dump(vars(args), jf, indent=2)
	except Exception:
		pass

	# Checkpoint helpers
	def latest_checkpoint_path():
		latest = os.path.join(ckpt_dir, "latest.pt")
		if os.path.exists(latest):
			return latest
		cands = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_step_*.pt")))
		return cands[-1] if cands else None

	def save_checkpoint(epoch_val: int, global_step_val: int) -> None:
		state = {
			"epoch": epoch_val,
			"global_step": global_step_val,
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scaler": (scaler.state_dict() if scaler is not None else None),
			"consecutive_epoch_correct": consecutive_epoch_correct,
			"tenth_epoch_streak_step": tenth_epoch_streak_step,
		}
		step_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step_val}.pt")
		torch.save(state, step_path)
		torch.save(state, os.path.join(ckpt_dir, "latest.pt"))

	# Try resume
	start_epoch = 1
	global_step = 0
	latest = latest_checkpoint_path()
	if latest:
		try:
			payload = torch.load(latest, map_location=device)
			model.load_state_dict(payload["model"])  # type: ignore[index]
			optimizer.load_state_dict(payload["optimizer"])  # type: ignore[index]
			if scaler is not None and payload.get("scaler") is not None:  # type: ignore[call-arg]
				scaler.load_state_dict(payload["scaler"])  # type: ignore[index]
			start_epoch = int(payload.get("epoch", 0)) + 1
			global_step = int(payload.get("global_step", 0))
			if "consecutive_epoch_correct" in payload:
				consecutive_epoch_correct.update(payload["consecutive_epoch_correct"])  # type: ignore[index]
			if "tenth_epoch_streak_step" in payload:
				tenth_epoch_streak_step.update(payload["tenth_epoch_streak_step"])  # type: ignore[index]
			print(f"Resumed from {latest} at epoch {start_epoch} global_step {global_step}")
		except Exception as e:
			print(f"Warning: failed to resume from {latest}: {e}")

	num_steps_per_epoch = len(train_loader)
	for epoch in range(start_epoch, args.epochs + 1):
		print(f"Epoch {epoch}/{args.epochs}")
		model.train()
		# Track if an example was correct at least once during this epoch
		epoch_correct_this_epoch: Dict[str, bool] = {p: False for p in train_paths_full}
		for step, (images, targets, batch_paths) in enumerate(train_loader, start=1):
			step_start = time.time()
			global_step += 1
			images = images.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)
			if scaler is not None:
				with torch.cuda.amp.autocast():
					logits = model(images)
					loss = criterion(logits, targets)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				logits = model(images)
				loss = criterion(logits, targets)
				loss.backward()
				optimizer.step()

			with torch.no_grad():
				preds = torch.argmax(logits, dim=1)
				correct_mask_batch = (preds == targets).cpu().tolist()
				batch_acc = float(sum(correct_mask_batch)) / max(1, len(correct_mask_batch))
				# Update per-epoch correctness flags
				for p, is_corr in zip(batch_paths, correct_mask_batch):
					if is_corr:
						epoch_correct_this_epoch[p] = True

			# Verbose per-step logging
			lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
			step_time = time.time() - step_start
			imgs_per_sec = int(images.size(0) / max(step_time, 1e-8))
			print(
				f"epoch={epoch} step={step}/{num_steps_per_epoch} global_step={global_step} "
				f"loss={float(loss.detach().item()):.4f} batch_acc={batch_acc:.4f} lr={lr:.6g} "
				f"time={step_time:.3f}s ips={imgs_per_sec}/s",
				flush=True,
			)

			# RIGHT/WRONG full-path logging
			right_paths = [p for p, c in zip(batch_paths, correct_mask_batch) if c]
			wrong_paths = [p for p, c in zip(batch_paths, correct_mask_batch) if not c]
			if right_paths:
				print("RIGHT: " + " | ".join(right_paths))
			if wrong_paths:
				print("WRONG: " + " | ".join(wrong_paths))

			# Record stats
			try:
				with open(stats_csv, "a", newline="", encoding="utf-8") as f:
					w = csv.writer(f)
					for p, is_corr in zip(batch_paths, correct_mask_batch):
						# For v2 (streak threshold), we still log per-step correctness only
						w.writerow([global_step, p, int(is_corr)])
			except Exception:
				pass

			# Save checkpoint after each step
			save_checkpoint(epoch, global_step)

		# End of epoch: update consecutive epoch correctness and record streak step
		for p, was_correct in epoch_correct_this_epoch.items():
			if was_correct:
				consecutive_epoch_correct[p] += 1
			else:
				consecutive_epoch_correct[p] = 0
			if consecutive_epoch_correct[p] >= args.streak_epochs and tenth_epoch_streak_step[p] == -1:
				tenth_epoch_streak_step[p] = global_step

		# Epoch summary
		print(f"  epoch_end_loss={float(loss.detach().item()):.4f}")
		# Save checkpoint at epoch end as well
		save_checkpoint(epoch, global_step)

	# Final summary
	remaining = sum(1 for v in tenth_epoch_streak_step.values() if v == -1)
	print(f"Training done. Examples never correct: {remaining}")

	summary_csv = os.path.join(model_out_dir, "first_correct_summary.csv")
	with open(summary_csv, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["path", "first_correct_step"])  # Here: step when streak threshold was first reached; -1 if never
		for p, step in tenth_epoch_streak_step.items():
			w.writerow([p, step])

	# Optional: run same analysis util if available
	try:
		from training_gradient_evaluator.order_analysis import analyze_and_plot
		analyze_and_plot(summary_csv, imagenet_csv=args.examples_csv)
	except Exception as e:
		print(f"Warning: analysis step failed: {e}")


if __name__ == "__main__":
	main()


