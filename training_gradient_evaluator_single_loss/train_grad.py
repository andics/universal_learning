import argparse
import csv
import os, sys
from typing import Dict, List, Tuple
from pathlib import Path
import time
import json
import glob
import copy
import random

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
import matplotlib.pyplot as plt

from training_gradient_evaluator_single_loss.data import ImageNetWrongExamplesDataset, read_imagenet_paths, extract_synset_from_path
from training_gradient_evaluator_single_loss.engine.single_example_trainer import SingleExampleTrainer, SingleExampleConfig


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


def calculate_weight_distance(initial_weights: Dict[str, torch.Tensor], final_weights: Dict[str, torch.Tensor]) -> float:
	"""Calculate Euclidean distance between two weight dictionaries.
	
	Args:
		initial_weights: Dictionary of parameter name -> initial weight tensor
		final_weights: Dictionary of parameter name -> final weight tensor
		
	Returns:
		Euclidean distance between flattened weight vectors
	"""
	distance_squared = 0.0
	for name in initial_weights:
		if name in final_weights:
			# Flatten tensors and compute squared difference
			initial_flat = initial_weights[name].flatten().float()
			final_flat = final_weights[name].flatten().float()
			diff = final_flat - initial_flat
			distance_squared += torch.sum(diff * diff).item()
	
	return float(distance_squared ** 0.5)


def _total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
	"""TV distance between categorical distributions p and q.
	This equals the 1-Wasserstein distance under a 0-1 ground metric."""
	p = np.asarray(p, dtype=np.float64)
	q = np.asarray(q, dtype=np.float64)
	p = p / (p.sum() + 1e-12)
	q = q / (q.sum() + 1e-12)
	return float(0.5 * np.abs(p - q).sum())


def _wasserstein_equal_bins_1d(p: np.ndarray, q: np.ndarray) -> float:
	"""Compute 1D W1 between discrete distributions on equally spaced bins (width=1)."""
	p = np.asarray(p, dtype=np.float64)
	q = np.asarray(q, dtype=np.float64)
	p = p / (p.sum() + 1e-12)
	q = q / (q.sum() + 1e-12)
	cdf_p = np.cumsum(p)
	cdf_q = np.cumsum(q)
	return float(np.abs(cdf_p - cdf_q).sum())


def _get_param_buckets(model: nn.Module) -> List[str]:
	"""Return stable parameter buckets by first name token (before first dot)."""
	buckets: List[str] = []
	seen = set()
	for name, _ in model.named_parameters():
		prefix = name.split('.')[0] if '.' in name else name
		if prefix not in seen:
			seen.add(prefix)
			buckets.append(prefix)
	return buckets


def _compute_grad_mass_distribution(model: nn.Module, buckets: List[str]) -> np.ndarray:
	"""Aggregate mean |grad| per provided bucket list and normalize to a distribution."""
	from collections import defaultdict
	agg = defaultdict(float)
	for name, p in model.named_parameters():
		if p.grad is None:
			continue
		prefix = name.split('.')[0] if '.' in name else name
		g = p.grad.detach()
		mass = float(g.abs().mean().item())
		agg[prefix] += mass
	masses = np.array([agg.get(b, 0.0) for b in buckets], dtype=np.float64)
	den = masses.sum()
	if den <= 0:
		return np.full_like(masses, 1.0 / len(masses)) if len(masses) > 0 else np.array([1.0])
	return masses / den


def read_imagenet_difficulty_order(csv_path: str) -> List[str]:
	"""Read the imagenet_examples_ammended.csv which contains paths in order of difficulty (easiest first)."""
	with open(csv_path, "r", encoding="utf-8") as f:
		text = f.read()
	text = text.lstrip("\ufeff").strip()
	if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
		text = text[1:-1]
	paths = [p.strip() for p in text.split(",") if p.strip()]
	return paths


def train_single_example(model: nn.Module, example_path: str, synset_to_idx: Dict[str, int], 
                       device: torch.device, train_tfms, optimizer: torch.optim.Optimizer, 
                       criterion: nn.Module, scaler, logger, max_steps: int = 1000, 
                       epsilon: float = 1e-6, csv_writer=None, initial_weights=None) -> Tuple[int, float, float, float, float, float]:
	"""Train on a single example until loss reaches epsilon or max_steps reached.
	
	Returns (total_steps, loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1) tuple. 
	If never reached epsilon, returns (-1, loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1).
	"""
	from PIL import Image
	
	# Load and prepare the single image
	image = Image.open(example_path).convert("RGB")
	x = train_tfms(image).unsqueeze(0).to(device)  # Add batch dimension

	# Softmax BEFORE training (P0)
	with torch.no_grad():
		model.eval()
		logits0 = model(x)
		probs0 = torch.softmax(logits0, dim=-1).squeeze(0).detach().cpu().numpy()
	
	# Get the target label
	wnid = extract_synset_from_path(example_path)
	if wnid is None or wnid not in synset_to_idx:
		raise RuntimeError(f"Could not determine synset/class for path: {example_path}")
	target = torch.tensor([synset_to_idx[wnid]], device=device)
	
	# Set model to train mode 
	model.train()
	
	# For single example training, we'll handle BatchNorm by setting it to eval mode
	# This uses pretrained running statistics instead of computing batch stats
	bn_modules = []
	for module in model.modules():
		if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
			bn_modules.append(module)
			module.eval()  # Use running stats to avoid batch size=1 error
	
	total_loss_sum = 0.0
	first_grad_mass = None
	last_grad_mass = None
	param_buckets = _get_param_buckets(model)
	
	for step in range(1, max_steps + 1):
		optimizer.zero_grad(set_to_none=True)
		
		if scaler is not None:
			with torch.amp.autocast('cuda'):
				logits = model(x)
				loss = criterion(logits, target)
			scaler.scale(loss).backward()
			# Unscale gradients for clipping
			scaler.unscale_(optimizer)
			# Capture gradient mass BEFORE stepping
			gmass = _compute_grad_mass_distribution(model, param_buckets)
			if first_grad_mass is None:
				first_grad_mass = gmass
			last_grad_mass = gmass
			# Clip gradients to prevent exploding gradients
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
		else:
			logits = model(x)
			loss = criterion(logits, target)
			loss.backward()
			# Capture gradient mass BEFORE stepping
			gmass = _compute_grad_mass_distribution(model, param_buckets)
			if first_grad_mass is None:
				first_grad_mass = gmass
			last_grad_mass = gmass
			# Clip gradients to prevent exploding gradients
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
		
		current_loss = float(loss.item())
		total_loss_sum += current_loss
		
		# Log each step to CSV if writer provided
		if csv_writer is not None:
			csv_writer.writerow([step, current_loss, total_loss_sum])
		
		# Check for NaN loss and stop if detected
		if torch.isnan(loss):
			# Calculate weight distance before returning
			if initial_weights is not None:
				final_weights = {name: param.data.clone().cpu() for name, param in model.named_parameters()}
				weight_distance = calculate_weight_distance(initial_weights, final_weights)
			else:
				weight_distance = 0.0
			# Compute softmax AFTER training (P1)
			with torch.no_grad():
				model.eval()
				logitsT = model(x)
				probsT = torch.softmax(logitsT, dim=-1).squeeze(0).detach().cpu().numpy()
			softmax_w1 = _total_variation_distance(probs0, probsT)
			# Grad-mass W1
			grad_mass_w1 = _wasserstein_equal_bins_1d(first_grad_mass if first_grad_mass is not None else np.array([1.0]), last_grad_mass if last_grad_mass is not None else np.array([1.0]))
			# Restore BatchNorm to train mode before returning
			for module in bn_modules:
				module.train()
			logger.warning(f"NaN loss detected at step {step}. Stopping training for this example.")
			return -1, total_loss_sum, float('nan'), weight_distance, softmax_w1, grad_mass_w1
		
		# Check if loss is within epsilon of zero
		if current_loss <= epsilon:
			# Calculate weight distance
			if initial_weights is not None:
				final_weights = {name: param.data.clone().cpu() for name, param in model.named_parameters()}
				weight_distance = calculate_weight_distance(initial_weights, final_weights)
			else:
				weight_distance = 0.0
			# Compute softmax AFTER training (P1)
			with torch.no_grad():
				model.eval()
				logitsT = model(x)
				probsT = torch.softmax(logitsT, dim=-1).squeeze(0).detach().cpu().numpy()
			softmax_w1 = _total_variation_distance(probs0, probsT)
			# Grad-mass W1
			grad_mass_w1 = _wasserstein_equal_bins_1d(first_grad_mass if first_grad_mass is not None else np.array([1.0]), last_grad_mass if last_grad_mass is not None else np.array([1.0]))
			# Restore BatchNorm to train mode
			for module in bn_modules:
				module.train()
			logger.info(f"Example {example_path} reached epsilon at step {step}, loss sum: {total_loss_sum:.4f}, final loss: {current_loss:.8f}, weight distance: {weight_distance:.4f}")
			return step, total_loss_sum, current_loss, weight_distance, softmax_w1, grad_mass_w1
		
		if step % 100 == 0:
			logger.info(f"Step {step}/{max_steps}, current loss: {current_loss:.8f}, loss sum: {total_loss_sum:.4f}")
	
	# Calculate final weight distance
	if initial_weights is not None:
		final_weights = {name: param.data.clone().cpu() for name, param in model.named_parameters()}
		weight_distance = calculate_weight_distance(initial_weights, final_weights)
	else:
		weight_distance = 0.0

	# Compute softmax AFTER training (P1)
	with torch.no_grad():
		model.eval()
		logitsT = model(x)
		probsT = torch.softmax(logitsT, dim=-1).squeeze(0).detach().cpu().numpy()
	softmax_w1 = _total_variation_distance(probs0, probsT)
	# Grad-mass W1
	grad_mass_w1 = _wasserstein_equal_bins_1d(first_grad_mass if first_grad_mass is not None else np.array([1.0]), last_grad_mass if last_grad_mass is not None else np.array([1.0]))
	
	# Restore BatchNorm to train mode before returning
	for module in bn_modules:
		module.train()
	
	logger.info(f"Example {example_path} never reached epsilon after {max_steps} steps, loss sum: {total_loss_sum:.4f}, final loss: {current_loss:.8f}, weight distance: {weight_distance:.4f}")
	return -1, total_loss_sum, current_loss, weight_distance, softmax_w1, grad_mass_w1


def main() -> None:
	parser = argparse.ArgumentParser(description="Train model on single examples in order of difficulty.")
	parser.add_argument("--model_name", type=str, default="efficientvit_m3.r224_in1k")
	parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "imagenet.npy"))
	parser.add_argument("--examples_csv", type=str, default=os.path.join("bars", "imagenet_examples_ammended.csv"))
	parser.add_argument("--root_dir", type=str, default=None)
	parser.add_argument("--model_csv_name", type=str, default="efficientvit_medium_3_224_classification_imagenet_1k",
						help="Model name to look up in imagenet_models.csv to select row in imagenet.npy")
	parser.add_argument("--imagenet_models_csv", type=str, default=os.path.join("bars", "imagenet_models.csv"),
						help="Path to bars/imagenet_models.csv containing model column names")
	parser.add_argument("--max_examples", type=int, default=500, help="Maximum number of examples to train on")
	parser.add_argument("--max_steps_per_example", type=int, default=10000, help="Maximum steps to train each example")
	parser.add_argument("--lr", type=float, default=0.001)
	parser.add_argument("--weight_decay", type=float, default=0)
	parser.add_argument("--epsilon", type=float, default=1e-3, help="Train until loss reaches this epsilon (default: 1e-6)")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--output_dir", type=str, default=os.path.join("training_gradient_evaluator_single_loss", "outputs"))
	parser.add_argument("--no_amp", action="store_true")
	parser.add_argument("--hierarchy_json", type=str, default=_default_hierarchy_json_path(), 
	                   help="Path to bars/imagenet_synset_hierarchy.json")
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	safe_model_name = args.model_name.replace('/', '_')
	model_out_dir = os.path.join(args.output_dir, safe_model_name)
	os.makedirs(model_out_dir, exist_ok=True)
	device = torch.device(args.device)

	# Configure timestamped logger in model output directory
	logger = logging.getLogger(f"train_grad_single_{safe_model_name}")
	logger.setLevel(logging.INFO)
	logger.propagate = False
	for h in list(logger.handlers):
		logger.removeHandler(h)
	from time import strftime, localtime
	stamp = strftime("%Y%m%d_%H%M%S", localtime())
	log_path = os.path.join(model_out_dir, f"train_single_{stamp}.log")
	fh = logging.FileHandler(log_path)
	fh.setLevel(logging.DEBUG)
	sh = logging.StreamHandler()
	sh.setLevel(logging.INFO)
	fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	fh.setFormatter(fmt)
	sh.setFormatter(fmt)
	logger.addHandler(fh)
	logger.addHandler(sh)

	# No global excepthook; rely on explicit try/except below to stop on errors and log tracebacks

	# Read paths in difficulty order (easiest first)
	difficulty_ordered_paths = read_imagenet_difficulty_order(args.examples_csv)
	logger.info(f"Loaded {len(difficulty_ordered_paths)} examples in difficulty order")

	# Build model (TIMM) and transforms
	import timm
	model = timm.create_model(args.model_name, pretrained=True)
	try:
		pcfg = getattr(model, 'pretrained_cfg', {}) or {}
		url = pcfg.get('url', None)
		hf_id = pcfg.get('hf_hub_id', None)
		logger.info(f"Loaded TIMM pretrained weights for {args.model_name}")
		logger.info(f"pretrained_cfg.url={url} hf_hub_id={hf_id}")
	except Exception as _e:
		logger.info(f"Note: could not display pretrained cfg details: {_e}")
	
	model = model.to(device)
	# Note: DataParallel disabled for single example training to avoid batch size issues
	# if torch.cuda.device_count() > 1 and device.type == "cuda":
	# 	model = nn.DataParallel(model)

	# Store original model weights for reset
	original_state_dict = copy.deepcopy(model.state_dict())
	logger.info("Stored original model weights for reset")

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
	logger.info(f"Resolved model_csv_name='{args.model_csv_name}' to mask_row_index={resolved_mask_row_index}")

	# Load mask to find wrong examples
	mask = np.load(args.bars_npy)
	if mask.ndim != 2 or resolved_mask_row_index < 0 or resolved_mask_row_index >= mask.shape[0]:
		raise ValueError(f"Unexpected mask shape {mask.shape} or bad row {resolved_mask_row_index}")
	correct_mask = mask[resolved_mask_row_index].astype(bool)
	wrong_mask = ~correct_mask
	wrong_indices = np.nonzero(wrong_mask)[0].tolist()
	
	# Find wrong examples that exist in the difficulty order
	def resolve_full(p: str) -> str:
		return os.path.join(args.root_dir, p) if args.root_dir and not os.path.isabs(p) else p

	# Get wrong examples that exist and map them to difficulty order
	all_paths = read_imagenet_paths(args.examples_csv)
	
	# Find wrong examples that exist 
	wrong_examples_with_difficulty = []
	path_to_difficulty_rank = {path: i for i, path in enumerate(difficulty_ordered_paths)}
	
	for idx in wrong_indices:
		if idx < len(all_paths):
			path = all_paths[idx]
			full_path = resolve_full(path)
			if os.path.exists(full_path) and path in path_to_difficulty_rank:
				difficulty_rank = path_to_difficulty_rank[path]
				wrong_examples_with_difficulty.append((path, difficulty_rank))
	
	logger.info(f"Found {len(wrong_examples_with_difficulty)} wrong examples that exist and have difficulty rankings")
	
	# Randomly sample from wrong examples
	selected_list_path = os.path.join(model_out_dir, "selected_examples.json")
	if os.path.exists(selected_list_path):
		with open(selected_list_path, 'r', encoding='utf-8') as sf:
			try:
				persisted = json.load(sf)
				if isinstance(persisted, list) and all(isinstance(t, list) and len(t) == 2 for t in persisted):
					selected_examples = [(str(p), int(rank)) for p, rank in persisted]
				else:
					selected_examples = wrong_examples_with_difficulty
			except Exception:
				selected_examples = wrong_examples_with_difficulty
	else:
		if len(wrong_examples_with_difficulty) > args.max_examples:
			random.seed(42)  # For reproducibility
			selected_examples = random.sample(wrong_examples_with_difficulty, args.max_examples)
		else:
			selected_examples = wrong_examples_with_difficulty
		with open(selected_list_path, 'w', encoding='utf-8') as sf:
			json.dump([[p, int(rank)] for p, rank in selected_examples], sf, indent=2)
	
	# Sort selected examples by difficulty (easiest first) for training order
	selected_examples.sort(key=lambda x: x[1])
	wrong_examples_ordered = [path for path, _ in selected_examples]
	
	logger.info(f"Selected {len(wrong_examples_ordered)} random wrong examples for training (sorted by difficulty)")

	# Prepare CSV for results
	results_csv = os.path.join(model_out_dir, "single_example_results.csv")
	csv_exists = os.path.exists(results_csv)
	if not csv_exists:
		with open(results_csv, "w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["example_index", "path", "total_steps_to_epsilon", "total_loss_sum", "final_loss", "weight_distance", "softmax_wasserstein", "grad_mass_wasserstein", "universal_difficulty_rank"])
	else:
		# Migrate header if missing new columns
		try:
			with open(results_csv, 'r', encoding='utf-8') as rf:
				reader = csv.reader(rf)
				header = next(reader, None)
				rows = list(reader)
			if header is not None and ("softmax_wasserstein" not in header or "grad_mass_wasserstein" not in header):
				new_header = ["example_index", "path", "total_steps_to_epsilon", "total_loss_sum", "final_loss", "weight_distance", "softmax_wasserstein", "grad_mass_wasserstein", "universal_difficulty_rank"]
				with open(results_csv, 'w', newline='', encoding='utf-8') as wf:
					writer = csv.writer(wf)
					writer.writerow(new_header)
					for row in rows:
						writer.writerow(row)
		except Exception:
			pass

	# Prepare directory for detailed step logs
	step_logs_dir = os.path.join(model_out_dir, "step_logs")
	os.makedirs(step_logs_dir, exist_ok=True)

	# Train each example individually
	results: List[Tuple[int, str, int, float, float, float, float, float, int]] = []

	# Build SingleExampleTrainer once
	from training_gradient_evaluator_single_loss.engine.single_example_trainer import _get_param_buckets  # noqa: F401 (ensure import path is valid)
	trainer = SingleExampleTrainer(model, synset_to_idx, train_tfms, logger)
	se_config = SingleExampleConfig(lr=args.lr, weight_decay=args.weight_decay, max_steps=int(args.max_steps_per_example), epsilon=float(args.epsilon), device=args.device, use_amp=(not args.no_amp))

	# Load already-processed paths for resume
	processed_paths: set[str] = set()
	if os.path.exists(results_csv):
		try:
			with open(results_csv, 'r', encoding='utf-8') as rf:
				reader = csv.reader(rf)
				header = next(reader, None)
				path_idx = None
				if header is not None:
					for j, h in enumerate(header):
						if h.strip().lower() == 'path':
							path_idx = j
							break
					if path_idx is not None:
						for row in reader:
							if len(row) > path_idx:
								processed_paths.add(row[path_idx])
		except Exception:
			pass

	for example_idx, example_path in enumerate(wrong_examples_ordered):
		if example_path in processed_paths:
			logger.info(f"Skipping already processed example {example_idx}: {example_path}")
			continue
		logger.info(f"\n=== Training Example {example_idx + 1}/{len(wrong_examples_ordered)}: {example_path} ===")
		
		# Create step-by-step log file for this example
		safe_path = example_path.replace('/', '_').replace('\\', '_').replace(':', '_')
		step_log_file = os.path.join(step_logs_dir, f"example_{example_idx}_{safe_path}_steps.csv")
		
		# Train on this single example (wrapped with try/except to log and stop on errors)
		full_path = resolve_full(example_path)
		try:
			initial_state = {name: param.data.clone().cpu() for name, param in model.named_parameters()}
			total_steps, total_loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1 = trainer.train_on_example(
				full_path, se_config, initial_state, step_log_file
			)
		except Exception as _e:
			import traceback
			logger.error("Fatal error while training on example", exc_info=True)
			raise
		
		# Get the actual universal difficulty rank (1-based)
		universal_rank = path_to_difficulty_rank[example_path] + 1
		results.append((example_idx, example_path, total_steps, total_loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1, universal_rank))
		
		# Append to CSV
		with open(results_csv, "a", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow([example_idx, example_path, total_steps, total_loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1, universal_rank])
		
		logger.info(f"Example {example_idx + 1} completed: {total_steps} steps, loss sum: {total_loss_sum:.4f}, final loss: {final_loss:.8f}, weight distance: {weight_distance:.4f} (universal rank: {universal_rank})")

	# Create final plots (use accumulated CSV to support resumed runs)
	logger.info("Creating final plots...")
	try:
		with open(results_csv, 'r', encoding='utf-8') as rf:
			reader = csv.reader(rf)
			header = next(reader, None)
			if not header:
				raise RuntimeError("Results CSV has no header")
			name_to_idx = {name: i for i, name in enumerate(header or [])}
			rows = [row for row in reader if row and len(row) == len(header)]
			# Accept rows even if they have more/less columns than the header (for backward compatibility)
			if not rows:
				rf.seek(0)
				reader = csv.reader(rf)
				_ = next(reader, None)
				rows = [row for row in reader if row]
			steps_idx = name_to_idx.get("total_steps_to_epsilon")
			rank_idx = name_to_idx.get("universal_difficulty_rank")
			loss_sum_idx = name_to_idx.get("total_loss_sum")
			weight_idx = name_to_idx.get("weight_distance")
			softmax_idx = name_to_idx.get("softmax_wasserstein")
			gradmass_idx = name_to_idx.get("grad_mass_wasserstein")
			X_steps, X_loss, X_weight, X_soft, X_gradmass, Y_rank = [], [], [], [], [], []
			for row in rows:
				try:
					st = int(float(row[steps_idx])) if steps_idx is not None else -1
					rk = int(float(row[rank_idx])) if rank_idx is not None else None
					if rk is None or st <= 0:
						continue
					X_steps.append(st)
					X_loss.append(float(row[loss_sum_idx]))
					X_weight.append(float(row[weight_idx]))
					Y_rank.append(rk)
					if softmax_idx is not None and row[softmax_idx] != '':
						X_soft.append(float(row[softmax_idx]))
					if gradmass_idx is not None and row[gradmass_idx] != '':
						X_gradmass.append(float(row[gradmass_idx]))
				except Exception:
					continue
			if len(Y_rank) >= 2:
				# Steps vs Difficulty
				plt.figure(figsize=(10, 6))
				plt.scatter(X_steps, Y_rank, alpha=0.7, s=50)
				plt.xlabel("Total SGD Steps to Reach Epsilon")
				plt.ylabel("Universal Difficulty Ranking (1=easiest)")
				plt.title(f"Universal Difficulty vs SGD Steps to Reach Epsilon\n({len(Y_rank)} examples, ε={args.epsilon})")
				plt.grid(True, alpha=0.3)
				corr = np.corrcoef(X_steps, Y_rank)[0, 1] if len(Y_rank) > 1 else 0.0
				plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
				plt.savefig(os.path.join(model_out_dir, "steps_vs_difficulty.png"), dpi=150, bbox_inches='tight')
				plt.close()
				# Loss vs Difficulty
				plt.figure(figsize=(10, 6))
				plt.scatter(X_loss, Y_rank, alpha=0.7, s=50, color='red')
				plt.xlabel("Total Loss Sum to Reach Epsilon")
				plt.ylabel("Universal Difficulty Ranking (1=easiest)")
				plt.title(f"Universal Difficulty vs Loss Sum to Reach Epsilon\n({len(Y_rank)} examples, ε={args.epsilon})")
				plt.grid(True, alpha=0.3)
				corr = np.corrcoef(X_loss, Y_rank)[0, 1] if len(Y_rank) > 1 else 0.0
				plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
				plt.savefig(os.path.join(model_out_dir, "loss_sum_vs_difficulty.png"), dpi=150, bbox_inches='tight')
				plt.close()
				# Weight Distance vs Difficulty
				plt.figure(figsize=(10, 6))
				plt.scatter(X_weight, Y_rank, alpha=0.7, s=50, color='green')
				plt.xlabel("Weight Distance to Reach Epsilon")
				plt.ylabel("Universal Difficulty Ranking (1=easiest)")
				plt.title(f"Universal Difficulty vs Weight Distance to Reach Epsilon\n({len(Y_rank)} examples, ε={args.epsilon})")
				plt.grid(True, alpha=0.3)
				corr = np.corrcoef(X_weight, Y_rank)[0, 1] if len(Y_rank) > 1 else 0.0
				plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
				plt.savefig(os.path.join(model_out_dir, "weight_distance_vs_difficulty.png"), dpi=150, bbox_inches='tight')
				plt.close()
				# Softmax W1 vs Difficulty
				if len(X_soft) == len(Y_rank) and len(X_soft) >= 2:
					plt.figure(figsize=(10, 6))
					plt.scatter(X_soft, Y_rank, alpha=0.7, s=50, color='purple')
					plt.xlabel("Softmax Distribution W1 (pre vs post)")
					plt.ylabel("Universal Difficulty Ranking (1=easiest)")
					plt.title(f"Universal Difficulty vs Softmax W1\n({len(Y_rank)} examples)")
					plt.grid(True, alpha=0.3)
					corr = np.corrcoef(X_soft, Y_rank)[0, 1] if len(Y_rank) > 1 else 0.0
					plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
					plt.savefig(os.path.join(model_out_dir, "softmax_wasserstein_vs_difficulty.png"), dpi=150, bbox_inches='tight')
					plt.close()
				# Grad-Mass W1 vs Difficulty
				if len(X_gradmass) == len(Y_rank) and len(X_gradmass) >= 2:
					plt.figure(figsize=(10, 6))
					plt.scatter(X_gradmass, Y_rank, alpha=0.7, s=50, color='brown')
					plt.xlabel("Gradient-Mass W1 (first vs final)")
					plt.ylabel("Universal Difficulty Ranking (1=easiest)")
					plt.title(f"Universal Difficulty vs Gradient-Mass W1\n({len(Y_rank)} examples)")
					plt.grid(True, alpha=0.3)
					corr = np.corrcoef(X_gradmass, Y_rank)[0, 1] if len(Y_rank) > 1 else 0.0
					plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
					plt.savefig(os.path.join(model_out_dir, "grad_mass_wasserstein_vs_difficulty.png"), dpi=150, bbox_inches='tight')
					plt.close()
			if len(Y_rank) < 2:
				logger.warning("Not enough successful examples to create meaningful plots")
	except Exception as _e:
		logger.warning(f"Could not create plots from results CSV: {_e}")
	
	# Save summary (build from CSV to support resumed runs)
	summary_path = os.path.join(model_out_dir, "training_summary.json")
	summary = {"model_name": args.model_name, "epsilon": args.epsilon}
	try:
		with open(results_csv, 'r', encoding='utf-8') as rf:
			reader = csv.reader(rf)
			header = next(reader, None)
			name_to_idx = {name: i for i, name in enumerate(header or [])}
			rows = [row for row in reader if row]
			success_count = 0
			fail_count = 0
			results_out = []
			for row in rows:
				st = int(float(row[name_to_idx.get("total_steps_to_epsilon", 0)]))
				if st > 0:
					success_count += 1
				else:
					fail_count += 1
				datum = {
					"path": row[name_to_idx.get("path", 1)],
					"total_steps": st,
					"total_loss_sum": float(row[name_to_idx.get("total_loss_sum", 0)]),
					"final_loss": float(row[name_to_idx.get("final_loss", 0)]),
					"weight_distance": float(row[name_to_idx.get("weight_distance", 0)]),
					"rank": int(float(row[name_to_idx.get("universal_difficulty_rank", 0)]))
				}
				if "softmax_wasserstein" in name_to_idx:
					datum["softmax_wasserstein"] = float(row[name_to_idx["softmax_wasserstein"]]) if row[name_to_idx["softmax_wasserstein"]] != '' else None
				if "grad_mass_wasserstein" in name_to_idx:
					datum["grad_mass_wasserstein"] = float(row[name_to_idx["grad_mass_wasserstein"]]) if row[name_to_idx["grad_mass_wasserstein"]] != '' else None
				results_out.append(datum)
			summary.update({
				"total_examples_attempted": len(rows),
				"successful_examples": success_count,
				"failed_examples": fail_count,
				"results": results_out
			})
	except Exception:
		# Fallback to this session's results only
		successful_results = [(idx, path, steps, loss_sum, final_loss, weight_dist, sm_w1, gm_w1, rank) for idx, path, steps, loss_sum, final_loss, weight_dist, sm_w1, gm_w1, rank in results if steps > 0]
		summary.update({
			"total_examples_attempted": len(results),
			"successful_examples": len(successful_results),
			"failed_examples": len(results) - len(successful_results),
			"results": [{
				"path": path,
				"total_steps": steps,
				"total_loss_sum": loss_sum,
				"final_loss": final_loss,
				"weight_distance": weight_dist,
				"softmax_wasserstein": sm_w1,
				"grad_mass_wasserstein": gm_w1,
				"rank": rank
			} for _, path, steps, loss_sum, final_loss, weight_dist, sm_w1, gm_w1, rank in results]
		})

	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)
	
	logger.info(f"Training complete. Summary saved to {summary_path}")
	logger.info(f"Results CSV: {results_csv}")


if __name__ == "__main__":
	main()