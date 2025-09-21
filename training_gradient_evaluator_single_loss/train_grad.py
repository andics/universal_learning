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
except Exception as _e:
	print(f"[WARN] Failed to set working dir/sys.path to Programming root: {_e}", file=sys.stderr)

import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from training_gradient_evaluator_single_loss.data import ImageNetWrongExamplesDataset, read_imagenet_paths, extract_synset_from_path
from training_gradient_evaluator_single_loss.engine.single_example_trainer import SingleExampleTrainer, SingleExampleConfig
from training_gradient_evaluator_single_loss.engine.selection import ExampleSelector
from training_gradient_evaluator_single_loss.engine.results import ResultsWriter
from training_gradient_evaluator_single_loss.utils.imagenet import default_hierarchy_json_path as _default_hierarchy_json_path, load_imagenet_hierarchy, read_imagenet_difficulty_order
from training_gradient_evaluator_single_loss.engine.cka import CKAManager


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


def main() -> None:
	parser = argparse.ArgumentParser(description="Train model on single examples in order of difficulty.")
	parser.add_argument("--model_name", type=str, default="efficientvit_b0.r224_in1k")
	parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "geq6wrong_21017_geq6correct_1525_imagenet.npy"))
	parser.add_argument("--examples_csv", type=str, default=os.path.join("bars", "geq6wrong_21017_geq6correct_1525_imagenet_examples_ammended.csv"))
	parser.add_argument("--root_dir", type=str, default=None)
	parser.add_argument("--model_csv_name", type=str, default="efficientvit_base_0_224_classification_imagenet_1k",
						help="Model name to look up in imagenet_models.csv to select row in imagenet.npy")
	parser.add_argument("--imagenet_models_csv", type=str, default=os.path.join("bars", "imagenet_models.csv"),
						help="Path to bars/imagenet_models.csv containing model column names")
	parser.add_argument("--max_examples", type=int, default=500, help="Maximum number of examples to train on")
	parser.add_argument("--max_steps_per_example", type=int, default=10000, help="Maximum steps to train each example")
	parser.add_argument("--lr", type=float, default=0.001)
	parser.add_argument("--weight_decay", type=float, default=0)
	parser.add_argument("--epsilon", type=float, default=1e-3, help="Train until loss reaches this epsilon")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--output_dir", type=str, default=os.path.join("training_gradient_evaluator_single_loss", "outputs"))
	parser.add_argument("--no_amp", action="store_true", help="Disable AMP (include this flag to turn AMP off)")
	parser.add_argument("--amp_dtype", type=str, default=None, choices=["float16", "bfloat16"], help="AMP dtype to use when AMP is enabled")
	parser.add_argument("--seed", type=int, default=1337, help="Global RNG seed for Python/NumPy/Torch")
	parser.add_argument("--deterministic", action="store_true", help="Force PyTorch deterministic algorithms")
	# Fraction of parameterized layers to compute CKA on (default 5%)
	parser.add_argument("--cka_layer_fraction", type=float, default=0.05, help="Fraction of parameterized layers to sample for CKA hooks (0-1]. Default: 0.05 (5%)")
	# Always use zero augmentation via timm regardless of this flag; kept for backward compatibility
	parser.add_argument("--zero_aug_train", action="store_true", help="(Deprecated) Zero augmentation is always enforced via timm")
	parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping max norm (use <=0 to disable)")
	parser.add_argument("--hierarchy_json", type=str, default=_default_hierarchy_json_path(), 
	                   help="Path to bars/imagenet_synset_hierarchy.json")
	# Optional explicit training examples JSON (same structure as produced by dataset_processing script)
	parser.add_argument("--explicit_examples_for_training", type=str, default=None,
						help="Path to JSON with explicit examples to train on; if provided, selection is restricted to these examples")
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

	# Log all parsed args early for full reproducibility
	logger.info("Parsed arguments:")
	for k, v in sorted(vars(args).items()):
		logger.info(f"  {k} = {v}")

	# No global excepthook; rely on explicit try/except below to stop on errors and log tracebacks

	# Read paths in difficulty order (easiest first)
	difficulty_ordered_paths = read_imagenet_difficulty_order(args.examples_csv)
	logger.info(f"Loaded {len(difficulty_ordered_paths)} examples in difficulty order")

	# Ensure cuBLAS determinism if requested (must be set before cuBLAS handle creation)
	if args.deterministic:
		try:
			if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
				os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
		except Exception:
			logger.exception("Failed to set CUBLAS_WORKSPACE_CONFIG for determinism")

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
		logger.warning(f"Could not display pretrained cfg details: {_e}")
	
	model = model.to(device)
	# Note: DataParallel disabled for single example training to avoid batch size issues
	# if torch.cuda.device_count() > 1 and device.type == "cuda":
	# 	model = nn.DataParallel(model)

	# Store original model weights for reset
	original_state_dict = copy.deepcopy(model.state_dict())
	logger.info("Stored original model weights for reset")

	# Build wnid->index/name mapping from hierarchy JSON
	synset_to_idx, index_to_name, wnid_to_words = load_imagenet_hierarchy(args.hierarchy_json)

	# Set seeds and deterministic backends if requested
	try:
		import random as _random
		_random.seed(int(args.seed))
		np.random.seed(int(args.seed))
		torch.manual_seed(int(args.seed))
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(int(args.seed))
		if args.deterministic:
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
			torch.use_deterministic_algorithms(True, warn_only=True)
	except Exception:
		logger.exception("Failed to set deterministic seeds/backends")

	# Build transforms; enforce zero augmentation via timm only
	data_config = timm.data.resolve_model_data_config(model)
	try:
		train_tfms = timm.data.create_transform(
			**data_config,
			is_training=True,
			no_aug=True,
			hflip=0.0,
			vflip=0.0,
			color_jitter=0.0,
			auto_augment=None,
			re_prob=0.0,
		)
	except Exception:
		logger.exception("Failed to create zero-augmentation transforms via timm")
		raise

	# Build single-example trainer and config
	trainer = SingleExampleTrainer(model, synset_to_idx, train_tfms, logger)
	# Determine AMP dtype default: prefer bfloat16 if supported on CUDA
	_auto_amp_dtype = None
	if not bool(args.no_amp) and torch.cuda.is_available():
		try:
			if torch.cuda.is_bf16_supported():
				_auto_amp_dtype = "bfloat16"
			else:
				_auto_amp_dtype = "float16"
		except Exception:
			_auto_amp_dtype = "float16"
	if args.amp_dtype is not None:
		_auto_amp_dtype = args.amp_dtype

	se_config = SingleExampleConfig(
		lr=args.lr,
		weight_decay=args.weight_decay,
		max_steps=int(args.max_steps_per_example),
		epsilon=float(args.epsilon),
		device=args.device,
		use_amp=not bool(args.no_amp),
		gradient_clip_norm=(args.grad_clip_norm if args.grad_clip_norm and args.grad_clip_norm > 0 else None),
		amp_dtype=_auto_amp_dtype,
	)
	# Also log resolved training configuration
	logger.info("Training configuration:")
	logger.info(f"  lr = {se_config.lr}")
	logger.info(f"  weight_decay = {se_config.weight_decay}")
	logger.info(f"  max_steps = {se_config.max_steps}")
	logger.info(f"  epsilon = {se_config.epsilon}")
	logger.info(f"  device = {se_config.device}")
	logger.info(f"  use_amp = {se_config.use_amp}")
	logger.info(f"  amp_dtype = {se_config.amp_dtype}")
	logger.info(f"  gradient_clip_norm = {se_config.gradient_clip_norm}")
	reset_state = copy.deepcopy(model.state_dict())

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

	# Load mask to find wrong examples (be robust to object arrays saved with pickle)
	mask = np.load(args.bars_npy, allow_pickle=True)

	if mask.ndim != 2 or resolved_mask_row_index < 0 or resolved_mask_row_index >= mask.shape[0]:
		raise ValueError(f"Unexpected mask shape {mask.shape} or bad row {resolved_mask_row_index}")
	correct_mask = mask[resolved_mask_row_index]
	# Build a boolean wrong mask that treats None placeholders as non-selectable (False)
	wrong_mask = np.array([False if i is None else (not bool(i)) for i in correct_mask], dtype=bool)
	wrong_indices = np.nonzero(wrong_mask)[0].tolist()
	
	# Resolve wrong example list with selection manager

	def _load_explicit_examples(json_path: str, all_paths_list: List[str], difficulty_list: List[str]) -> set[str]:
		"""Load explicit examples from JSON; accept list of dicts with 'image_path' or 'path',
		list of strings (paths), or list of integers (ranks). Return a set of paths.
		"""
		with open(json_path, 'r', encoding='utf-8') as jf:
			data = json.load(jf)
		allowed: set[str] = set()
		if isinstance(data, list):
			for item in data:
				try:
					if isinstance(item, dict):
						p = item.get('image_path') or item.get('path')
						if isinstance(p, str):
							allowed.add(p)
							continue
						r = item.get('image_rank') if isinstance(item, dict) else None
						if isinstance(r, int) and 0 <= r < len(difficulty_list):
							allowed.add(difficulty_list[r])
							continue
					elif isinstance(item, str):
						allowed.add(item)
					elif isinstance(item, int):
						if 0 <= item < len(difficulty_list):
							allowed.add(difficulty_list[item])
				except Exception:
					continue
		# Keep only those present in difficulty-ordered list to ensure consistent ranking
		allowed &= set(difficulty_list)
		return allowed
	def resolve_full(p: str) -> str:
		return os.path.join(args.root_dir, p) if args.root_dir and not os.path.isabs(p) else p
	all_paths = read_imagenet_paths(args.examples_csv)
	selector = ExampleSelector(model_out_dir, seed=int(args.seed))

	# If an explicit examples JSON is provided, restrict candidate pool to those examples
	candidate_indices: List[int]
	if args.explicit_examples_for_training:
		try:
			allowed_paths = _load_explicit_examples(args.explicit_examples_for_training, all_paths, difficulty_ordered_paths)
			# Build indices into all_paths for allowed paths
			path_to_index: Dict[str, int] = {p: i for i, p in enumerate(all_paths)}
			candidate_indices = [path_to_index[p] for p in allowed_paths if p in path_to_index]
			logger.info(f"Restricting candidate pool to {len(candidate_indices)} explicit examples from JSON")
		except Exception as _e:
			logger.exception("Failed to load explicit_examples_for_training; falling back to default wrong-only selection")
			candidate_indices = wrong_indices
	else:
		candidate_indices = wrong_indices

	wrong_examples_ordered = selector.select_wrong_examples(candidate_indices, all_paths, difficulty_ordered_paths, args.root_dir, args.max_examples)
	logger.info(f"Selected {len(wrong_examples_ordered)} wrong examples for training (sorted by difficulty)")
	path_to_difficulty_rank = {path: i for i, path in enumerate(difficulty_ordered_paths)}
	# Output full list of rank indices that will be trained on
	train_indices = [path_to_difficulty_rank[p] for p in wrong_examples_ordered if p in path_to_difficulty_rank]
	try:
		logger.info("Training examples (rank indices, 0-based): " + ", ".join(str(int(i)) for i in train_indices))
		train_examples_csv = os.path.join(model_out_dir, "train_examples.csv")
		with open(train_examples_csv, 'w', newline='', encoding='utf-8') as tf:
			w = csv.writer(tf)
			w.writerow(["rank_index"])
			for idx in train_indices:
				w.writerow([int(idx)])
	except Exception as _e:
		logger.exception("Failed to write train_examples.csv or log training indices")
	
	# Results manager
	results = ResultsWriter(model_out_dir)
	results_csv = results.results_csv

	# Prepare directory for detailed step logs
	step_logs_dir = os.path.join(model_out_dir, "step_logs")
	os.makedirs(step_logs_dir, exist_ok=True)

	# Configure and initialize CKA manager
	cka_mgr = CKAManager(
		model=model,
		transform=train_tfms,
		device=device,
		layer_fraction=float(args.cka_layer_fraction),
		seed=int(args.seed),
		root_dir=args.root_dir,
		difficulty_paths=difficulty_ordered_paths,
		model_out_dir=model_out_dir,
		logger=logger,
		num_images=50,
		plot_interval_ranks=1000,
	)
	if float(args.cka_layer_fraction) <= 0.0:
		logger.info("CKA disabled: cka_layer_fraction == 0. Skipping all CKA computations.")
	else:
		cka_mgr.setup_baseline()

	# Train each example individually

	# Load already-processed paths for resume
	processed_paths = selector.load_processed_paths(results_csv)

	for example_idx, example_path in enumerate(wrong_examples_ordered):
		if example_path in processed_paths:
			logger.info(f"Skipping already processed example {example_idx}: {example_path}")
			continue
		logger.info(f"\n=== Training Example {example_idx + 1}/{len(wrong_examples_ordered)}: {example_path} ===")
		
		# Create step-by-step log file for this example, include rank and only parent dir + filename
		parent_name = os.path.basename(os.path.dirname(example_path))
		file_name = os.path.basename(example_path)
		short_id = f"{parent_name}_{file_name}" if parent_name else file_name
		# Sanitize just in case
		short_id = short_id.replace('/', '_').replace('\\', '_').replace(':', '_')
		_rank = path_to_difficulty_rank.get(example_path, -1)
		step_log_file = os.path.join(step_logs_dir, f"example_{example_idx}_rank_{_rank:05d}_{short_id}_steps.csv")
		
		# Train on this single example (wrapped with try/except to log and stop on errors)
		full_path = resolve_full(example_path)
		try:
			total_steps, total_loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1, init_highest_softmax_prob, init_target_softmax_prob, steps_to_correct = trainer.train_on_example(
				full_path, se_config, reset_state, step_log_file
			)
			# After training this one example, compute CKA artifacts via manager
			global_cka = cka_mgr.after_example(model, example_path, short_id, _rank)
		except Exception as _e:
			import traceback
			logger.error("Fatal error while training on example", exc_info=True)
			raise
		
		# Get the actual universal difficulty rank (1-based)
		universal_rank = path_to_difficulty_rank[example_path] + 1
		# Append to CSV once (including global CKA and new metrics)
		results.append([example_idx, example_path, total_steps, total_loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1, universal_rank, global_cka, init_highest_softmax_prob, init_target_softmax_prob, steps_to_correct])
		
		logger.info(f"Example {example_idx + 1} completed: {total_steps} steps, loss sum: {total_loss_sum:.4f}, final loss: {final_loss:.8f}, weight distance: {weight_distance:.4f} (universal rank: {universal_rank})")

	# Finalize: plots and summary
	logger.info("Creating final plots...")
	results.build_plots(epsilon=float(args.epsilon))
	# Always write correlations.json based on existing CSV (idempotent)
	try:
		results.write_correlations(overwrite=True)
	except Exception:
		logger.exception("Failed writing correlations.json; continuing")
	results.write_summary(epsilon=float(args.epsilon))
	summary_path = os.path.join(model_out_dir, "training_summary.json")
	logger.info(f"Training complete. Summary saved to {summary_path}")
	logger.info(f"Results CSV: {results_csv}")


if __name__ == "__main__":
	main()