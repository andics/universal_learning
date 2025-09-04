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
from training_gradient_evaluator_single_loss.engine.selection import ExampleSelector
from training_gradient_evaluator_single_loss.engine.results import ResultsWriter
from training_gradient_evaluator_single_loss.utils.imagenet import default_hierarchy_json_path as _default_hierarchy_json_path, load_imagenet_hierarchy, read_imagenet_difficulty_order


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

	# Build single-example trainer and config
	trainer = SingleExampleTrainer(model, synset_to_idx, train_tfms, logger)
	se_config = SingleExampleConfig(
		lr=args.lr,
		weight_decay=args.weight_decay,
		max_steps=int(args.max_steps_per_example),
		epsilon=float(args.epsilon),
		device=args.device,
		use_amp=(not args.no_amp),
	)

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
	
	# Resolve wrong example list with selection manager
	def resolve_full(p: str) -> str:
		return os.path.join(args.root_dir, p) if args.root_dir and not os.path.isabs(p) else p
	all_paths = read_imagenet_paths(args.examples_csv)
	selector = ExampleSelector(model_out_dir, seed=42)
	wrong_examples_ordered = selector.select_wrong_examples(wrong_indices, all_paths, difficulty_ordered_paths, args.root_dir, args.max_examples)
	logger.info(f"Selected {len(wrong_examples_ordered)} wrong examples for training (sorted by difficulty)")
	path_to_difficulty_rank = {path: i for i, path in enumerate(difficulty_ordered_paths)}
	
	# Results manager
	results = ResultsWriter(model_out_dir)
	results_csv = results.results_csv

	# Prepare directory for detailed step logs
	step_logs_dir = os.path.join(model_out_dir, "step_logs")
	os.makedirs(step_logs_dir, exist_ok=True)

	# Train each example individually
	
	# Load already-processed paths for resume
	processed_paths = selector.load_processed_paths(results_csv)

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
		# Append to CSV once
		results.append([example_idx, example_path, total_steps, total_loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1, universal_rank])
		
		logger.info(f"Example {example_idx + 1} completed: {total_steps} steps, loss sum: {total_loss_sum:.4f}, final loss: {final_loss:.8f}, weight distance: {weight_distance:.4f} (universal rank: {universal_rank})")

	# Finalize: plots and summary
	logger.info("Creating final plots...")
	results.build_plots(epsilon=float(args.epsilon))
	results.write_summary(epsilon=float(args.epsilon))
	summary_path = os.path.join(model_out_dir, "training_summary.json")
	logger.info(f"Training complete. Summary saved to {summary_path}")
	logger.info(f"Results CSV: {results_csv}")


if __name__ == "__main__":
	main()