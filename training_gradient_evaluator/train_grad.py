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
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training_gradient_evaluator.data import ImageNetWrongExamplesDataset, read_imagenet_paths, extract_synset_from_path


def filter_existing_indices(paths: List[str], indices: List[int], root_dir: str | None) -> List[int]:
	kept: List[int] = []
	for idx in indices:
		p = paths[idx]
		full = os.path.join(root_dir, p) if root_dir and not os.path.isabs(p) else p
		if os.path.exists(full):
			kept.append(idx)
	return kept


def load_wnid_to_index_from_torchvision() -> Dict[str, int] | None:
	"""Load the standard ImageNet-1k index mapping via torchvision's imagenet_class_index.json."""
	try:
		import torchvision
		idx_json = os.path.join(os.path.dirname(torchvision.__file__), 'datasets', 'imagenet_class_index.json')
		with open(idx_json, 'r', encoding='utf-8') as f:
			data = json.load(f)  # keys are str indices, values [wnid, classname]
		wnid_to_idx: Dict[str, int] = {}
		for k, v in data.items():
			try:
				i = int(k)
				wnid = str(v[0])
				wnid_to_idx[wnid] = i
			except Exception:
				continue
		return wnid_to_idx
	except Exception:
		return None


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
		if idx not in idx_to_words:
			idx_to_words[idx] = words
	return wnid_to_idx, idx_to_words, wnid_to_words


def main() -> None:
	parser = argparse.ArgumentParser(description="V2: Train model on only the images it originally got wrong; torchvision transforms.")
	parser.add_argument("--model_name", type=str, default="resnet34.a3_in1k")
	parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "imagenet.npy"))
	parser.add_argument("--examples_csv", type=str, default=os.path.join("bars", "imagenet_examples_ammended.csv"))
	# mapping_txt no longer used; hierarchy_json replaces it
	parser.add_argument("--root_dir", type=str, default=None)
	# Select the row in imagenet.npy by model name looked up from bars/imagenet_models.csv
	parser.add_argument("--model_csv_name", type=str, default="resnet_34_160_classification_imagenet_1k", help="Model name to look up in imagenet_models.csv to select row in imagenet.npy")
	parser.add_argument("--imagenet_models_csv", type=str, default=os.path.join("bars", "imagenet_models.csv"), help="Path to bars/imagenet_models.csv containing model column names")
	parser.add_argument("--epochs", type=int, default=8)
	parser.add_argument("--batch_size", type=int, default=1024)
	parser.add_argument("--lr", type=float, default=40e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-2)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--output_dir", type=str, default=os.path.join("training_gradient_evaluator", "outputs"))
	parser.add_argument("--no_amp", action="store_true")
	parser.add_argument("--streak_epochs", type=int, default=2, help="(Deprecated) Unused; first-correct computed at global-step granularity")
	parser.add_argument("--hierarchy_json", type=str, default=_default_hierarchy_json_path(), help="Path to bars/imagenet_synset_hierarchy.json")
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	safe_model_name = args.model_name.replace('/', '_')
	model_out_dir = os.path.join(args.output_dir, safe_model_name)
	os.makedirs(model_out_dir, exist_ok=True)
	ckpt_dir = os.path.join(model_out_dir, "checkpoints")
	os.makedirs(ckpt_dir, exist_ok=True)
	device = torch.device(args.device)

	# Configure timestamped logger in model output directory
	logger = logging.getLogger(f"train_grad_{safe_model_name}")
	logger.setLevel(logging.INFO)
	logger.propagate = False
	for h in list(logger.handlers):
		logger.removeHandler(h)
	from time import strftime, localtime
	stamp = strftime("%Y%m%d_%H%M%S", localtime())
	log_path = os.path.join(model_out_dir, f"train_{stamp}.log")
	fh = logging.FileHandler(log_path)
	fh.setLevel(logging.INFO)
	sh = logging.StreamHandler()
	sh.setLevel(logging.INFO)
	fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	fh.setFormatter(fmt)
	sh.setFormatter(fmt)
	logger.addHandler(fh)
	logger.addHandler(sh)

	# Paths
	paths = read_imagenet_paths(args.examples_csv)
	if not paths:
		raise RuntimeError(f"No image paths found in {args.examples_csv}")

	# Build model (TIMM) and transforms
	import timm
	model = timm.create_model(args.model_name, pretrained=True)
	try:
		pcfg = getattr(model, 'pretrained_cfg', {}) or {}
		url = pcfg.get('url', None)
		hf_id = pcfg.get('hf_hub_id', None)
		logger.info(f"Loaded TIMM pretrained weights for {args.model_name}")
		logger.info(f"pretrained_cfg.url={url} hf_hub_id={hf_id}")
		logger.info(f"torch.hub cache dir: {torch.hub.get_dir()}")
		from pathlib import Path as _P
		logger.info("HF caches: %s %s %s", os.getenv("HF_HOME"), os.getenv("HUGGINGFACE_HUB_CACHE"), str(_P.home() / ".cache/huggingface/hub"))
	except Exception as _e:
		logger.info(f"Note: could not display pretrained cfg details: {_e}")
	model = model.to(device)
	if torch.cuda.device_count() > 1 and device.type == "cuda":
		model = nn.DataParallel(model)

	# Build wnid->index/name mapping from hierarchy JSON
	synset_to_idx, index_to_name, wnid_to_words = load_imagenet_hierarchy(args.hierarchy_json)

	# Build training transforms from timm model data_config (train pipeline)
	data_config = timm.data.resolve_model_data_config(model)
	train_tfms = timm.data.create_transform(**data_config, is_training=True)

	# Resolve mask row index from CSV of model names
	def _find_model_index(models_csv_path: str, model_name: str) -> int:
		with open(models_csv_path, 'r', encoding='utf-8') as f:
			reader = csv.reader(f)
			for row in reader:
				for j, name in enumerate(row):
					if name.strip() == model_name:
						return j
		raise ValueError(f"Model '{model_name}' not found in {models_csv_path}")

	resolved_mask_row_index = _find_model_index(args.imagenet_models_csv, args.model_csv_name)
	logger.info(f"Resolved model_csv_name='{args.model_csv_name}' to mask_row_index={resolved_mask_row_index} using {args.imagenet_models_csv}")

	# Mask & wrong indices
	mask = np.load(args.bars_npy)
	if mask.ndim != 2 or resolved_mask_row_index < 0 or resolved_mask_row_index >= mask.shape[0]:
		raise ValueError(f"Unexpected mask shape {mask.shape} or bad row {resolved_mask_row_index}")
	correct_mask = mask[resolved_mask_row_index].astype(bool)
	wrong_mask = ~correct_mask
	wrong_indices = np.nonzero(wrong_mask)[0].tolist()
	wrong_indices = filter_existing_indices(paths, wrong_indices, args.root_dir)
	if not wrong_indices:
		raise RuntimeError("No existing images among wrong examples.")
	logger.info(f"Training on {len(wrong_indices)} originally-wrong examples.")

	# Dataset & loader with wnid-based label indices
	train_ds = ImageNetWrongExamplesDataset(paths, wrong_indices, synset_to_idx, transform=train_tfms, root_dir=args.root_dir)
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

	# Optimizer and AMP
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scaler = None if args.no_amp or device.type != "cuda" else torch.cuda.amp.GradScaler()

	# Stats logging
	def resolve_full(p: str) -> str:
		return os.path.join(args.root_dir, p) if args.root_dir and not os.path.isabs(p) else p

	train_paths_full = [resolve_full(paths[i]) for i in wrong_indices]
	stats_csv = os.path.join(model_out_dir, "example_statistics.csv")
	# Create file and header only if not present
	if not os.path.exists(stats_csv) or os.path.getsize(stats_csv) == 0:
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
		}
		step_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step_val}.pt")
		torch.save(state, step_path)
		torch.save(state, os.path.join(ckpt_dir, "latest.pt"))

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
			logger.info(f"Resumed from {latest} at epoch {start_epoch} global_step {global_step}")
		except Exception as e:
			logger.warning(f"Warning: failed to resume from {latest}: {e}")

	num_steps_per_epoch = len(train_loader)
	for epoch in range(start_epoch, args.epochs + 1):
		logger.info(f"Epoch {epoch}/{args.epochs}")
		model.train()
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
				# Correctness based on numeric class index from hierarchy
				correct_mask_batch: List[bool] = []
				for p, pred_idx in zip(batch_paths, preds.cpu().tolist()):
					wnid = extract_synset_from_path(p)
					tgt_idx = synset_to_idx.get(wnid, -1)
					ok = int(pred_idx) == int(tgt_idx)
					correct_mask_batch.append(ok)
					# per-step correctness is logged; first-correct will be computed from stats CSV
				batch_acc = float(sum(correct_mask_batch)) / max(1, len(correct_mask_batch))

			lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
			step_time = time.time() - step_start
			imgs_per_sec = int(images.size(0) / max(step_time, 1e-8))
			logger.info(
				f"epoch={epoch} step={step}/{num_steps_per_epoch} global_step={global_step} "
				f"loss={float(loss.detach().item()):.4f} batch_acc={batch_acc:.4f} lr={lr:.6g} "
				f"time={step_time:.3f}s ips={imgs_per_sec}/s"
			)

			right_paths = [p for p, c in zip(batch_paths, correct_mask_batch) if c]
			wrong_paths = [p for p, c in zip(batch_paths, correct_mask_batch) if not c]
			if right_paths:
				logger.info("RIGHT: " + " | ".join(right_paths))
			if wrong_paths:
				logger.info("WRONG: " + " | ".join(wrong_paths))

			try:
				with open(stats_csv, "a", newline="", encoding="utf-8") as f:
					w = csv.writer(f)
					for p, is_corr in zip(batch_paths, correct_mask_batch):
						w.writerow([global_step, p, int(is_corr)])
			except Exception as e:
				logger.warning(f"Failed to append stats: {e}")

			save_checkpoint(epoch, global_step)

		print(f"  epoch_end_loss={float(loss.detach().item()):.4f}")
		save_checkpoint(epoch, global_step)

	# Compute first-time-correct (global step) by scanning stats CSV
	first_correct: Dict[str, int] = {p: -1 for p in train_paths_full}
	try:
		with open(stats_csv, "r", encoding="utf-8") as f:
			r = csv.reader(f)
			head = next(r, None)
			for line in r:
				if len(line) < 3:
					continue
				try:
					step_val = int(line[0])
					corr_val = int(line[2])
				except Exception:
					continue
				path_val = line[1]
				if corr_val == 1 and path_val in first_correct and first_correct[path_val] == -1:
					first_correct[path_val] = step_val
	except Exception as e:
		logger.warning(f"Failed to parse stats for first-correct: {e}")

	remaining = sum(1 for v in first_correct.values() if v == -1)
	logger.info(f"Training done. Examples never correct: {remaining}")

	summary_csv = os.path.join(model_out_dir, "first_correct_summary.csv")
	with open(summary_csv, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["path", "first_correct_step"])  # step when first correct was observed; -1 if never
		for p, step in first_correct.items():
			w.writerow([p, step])

	try:
		from training_gradient_evaluator.order_analysis import analyze_and_plot
		analyze_and_plot(summary_csv, imagenet_csv=args.examples_csv)
	except Exception as e:
		logger.warning(f"Warning: analysis step failed: {e}")


if __name__ == "__main__":
	main()


