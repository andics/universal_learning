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

from training_gradient_evaluator_single.data import ImageNetWrongExamplesDataset, read_imagenet_paths, extract_synset_from_path


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
                        criterion: nn.Module, scaler, logger, max_steps: int = 1000) -> int:
	"""Train on a single example until it gets it right or max_steps reached.
	
	Returns the number of steps it took to get it right, or -1 if never got it right.
	"""
	from PIL import Image
	
	# Load and prepare the single image
	image = Image.open(example_path).convert("RGB")
	x = train_tfms(image).unsqueeze(0).to(device)  # Add batch dimension
	
	# Get the target label
	wnid = extract_synset_from_path(example_path)
	if wnid is None or wnid not in synset_to_idx:
		raise RuntimeError(f"Could not determine synset/class for path: {example_path}")
	target = torch.tensor([synset_to_idx[wnid]], device=device)
	
	model.train()
	for step in range(1, max_steps + 1):
		optimizer.zero_grad(set_to_none=True)
		
		if scaler is not None:
			with torch.cuda.amp.autocast():
				logits = model(x)
				loss = criterion(logits, target)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			logits = model(x)
			loss = criterion(logits, target)
			loss.backward()
			optimizer.step()
		
		# Check if correct
		with torch.no_grad():
			pred = torch.argmax(logits, dim=1)
			if pred.item() == target.item():
				logger.info(f"Example {example_path} got correct at step {step}")
				return step
		
		if step % 100 == 0:
			logger.info(f"Step {step}/{max_steps}, loss: {loss.item():.4f}")
	
	logger.info(f"Example {example_path} never got correct after {max_steps} steps")
	return -1


def main() -> None:
	parser = argparse.ArgumentParser(description="Train model on single examples in order of difficulty.")
	parser.add_argument("--model_name", type=str, default="resnet34.a3_in1k")
	parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "imagenet.npy"))
	parser.add_argument("--examples_csv", type=str, default=os.path.join("bars", "imagenet_examples_ammended.csv"))
	parser.add_argument("--root_dir", type=str, default=None)
	parser.add_argument("--model_csv_name", type=str, default="resnet_34_160_classification_imagenet_1k",
						help="Model name to look up in imagenet_models.csv to select row in imagenet.npy")
	parser.add_argument("--imagenet_models_csv", type=str, default=os.path.join("bars", "imagenet_models.csv"),
						help="Path to bars/imagenet_models.csv containing model column names")
	parser.add_argument("--max_examples", type=int, default=1000, help="Maximum number of examples to train on")
	parser.add_argument("--max_steps_per_example", type=int, default=1000, help="Maximum steps to train each example")
	parser.add_argument("--lr", type=float, default=5e-6)
	parser.add_argument("--weight_decay", type=float, default=0)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--output_dir", type=str, default=os.path.join("training_gradient_evaluator_single", "outputs"))
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
	fh.setLevel(logging.INFO)
	sh = logging.StreamHandler()
	sh.setLevel(logging.INFO)
	fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	fh.setFormatter(fmt)
	sh.setFormatter(fmt)
	logger.addHandler(fh)
	logger.addHandler(sh)

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
	if torch.cuda.device_count() > 1 and device.type == "cuda":
		model = nn.DataParallel(model)

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
	if len(wrong_examples_with_difficulty) > args.max_examples:
		random.seed(42)  # For reproducibility
		selected_examples = random.sample(wrong_examples_with_difficulty, args.max_examples)
	else:
		selected_examples = wrong_examples_with_difficulty
	
	# Sort selected examples by difficulty (easiest first) for training order
	selected_examples.sort(key=lambda x: x[1])
	wrong_examples_ordered = [path for path, _ in selected_examples]
	
	logger.info(f"Selected {len(wrong_examples_ordered)} random wrong examples for training (sorted by difficulty)")

	# Prepare CSV for results
	results_csv = os.path.join(model_out_dir, "single_example_results.csv")
	with open(results_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["example_index", "path", "steps_to_correct", "universal_difficulty_rank"])

	# Train each example individually
	criterion = nn.CrossEntropyLoss()
	results: List[Tuple[int, str, int, int]] = []
	
	for example_idx, example_path in enumerate(wrong_examples_ordered):
		logger.info(f"\n=== Training Example {example_idx + 1}/{len(wrong_examples_ordered)}: {example_path} ===")
		
		# Reset model to original weights
		model.load_state_dict(original_state_dict)
		logger.info("Reset model to original weights")
		
		# Create fresh optimizer for this example
		optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		scaler = None if args.no_amp or device.type != "cuda" else torch.cuda.amp.GradScaler()
		
		# Train on this single example
		full_path = resolve_full(example_path)
		steps_to_correct = train_single_example(
			model, full_path, synset_to_idx, device, train_tfms, 
			optimizer, criterion, scaler, logger, args.max_steps_per_example
		)
		
		# Get the actual universal difficulty rank (1-based)
		universal_rank = path_to_difficulty_rank[example_path] + 1
		results.append((example_idx, example_path, steps_to_correct, universal_rank))
		
		# Append to CSV
		with open(results_csv, "a", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow([example_idx, example_path, steps_to_correct, universal_rank])
		
		logger.info(f"Example {example_idx + 1} completed: {steps_to_correct} steps (universal rank: {universal_rank})")

	# Create final plot
	logger.info("Creating final plot...")
	
	# Filter out examples that never got correct
	successful_results = [(idx, path, steps, rank) for idx, path, steps, rank in results if steps > 0]
	
	if len(successful_results) >= 2:
		steps_to_correct = [steps for _, _, steps, _ in successful_results]
		difficulty_ranks = [rank for _, _, _, rank in successful_results]
		
		plt.figure(figsize=(10, 6))
		plt.scatter(steps_to_correct, difficulty_ranks, alpha=0.7, s=50)
		plt.xlabel("Steps to Get Correct")
		plt.ylabel("Universal Difficulty Ranking (1=easiest)")
		plt.title(f"Universal Difficulty vs Steps to Get Correct\n({len(successful_results)} examples)")
		plt.grid(True, alpha=0.3)
		
		# Add correlation info
		if len(successful_results) > 1:
			correlation = np.corrcoef(steps_to_correct, difficulty_ranks)[0, 1]
			plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
			        transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
		
		plot_path = os.path.join(model_out_dir, "steps_vs_difficulty.png")
		plt.savefig(plot_path, dpi=150, bbox_inches='tight')
		plt.close()
		logger.info(f"Saved plot to {plot_path}")
	else:
		logger.warning("Not enough successful examples to create meaningful plot")

	# Save summary
	summary_path = os.path.join(model_out_dir, "training_summary.json")
	summary = {
		"model_name": args.model_name,
		"total_examples_attempted": len(wrong_examples_ordered),
		"successful_examples": len(successful_results),
		"failed_examples": len(wrong_examples_ordered) - len(successful_results),
		"results": [{"path": path, "steps": steps, "rank": rank} 
		           for _, path, steps, rank in results]
	}
	
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)
	
	logger.info(f"Training complete. Summary saved to {summary_path}")
	logger.info(f"Results CSV: {results_csv}")


if __name__ == "__main__":
	main()