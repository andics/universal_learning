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

# Use local, self-contained copies of OOP components
from training_gradient_evaluator_single_loss_parallel.data import ImageNetWrongExamplesDataset, read_imagenet_paths, extract_synset_from_path
from training_gradient_evaluator_single_loss_parallel.engine.single_example_trainer import SingleExampleTrainer, SingleExampleConfig
from training_gradient_evaluator_single_loss_parallel.engine.selection import ExampleSelector
from training_gradient_evaluator_single_loss_parallel.engine.results import ResultsWriter
from training_gradient_evaluator_single_loss_parallel.utils.imagenet import default_hierarchy_json_path as _default_hierarchy_json_path, load_imagenet_hierarchy, read_imagenet_difficulty_order
from training_gradient_evaluator_single_loss_parallel.engine.cka_manager import CKAManager



def _compute_worker_slice(n_total: int, num_workers: int, current_worker: int) -> Tuple[int, int]:
	if num_workers <= 0:
		raise ValueError("num_workers must be >= 1")
	if current_worker < 0 or current_worker >= num_workers:
		raise ValueError("current_worker must be in [0, num_workers-1]")
	base = n_total // num_workers
	rem = n_total % num_workers
	start = current_worker * base + min(current_worker, rem)
	end = start + base + (1 if current_worker < rem else 0)
	return start, end


def main() -> None:
	parser = argparse.ArgumentParser(description="Parallel worker: Train model on assigned partition of single examples.")
	parser.add_argument("--model_name", type=str, default="efficientvit_b0.r224_in1k")
	parser.add_argument("--bars_npy", type=str, default=os.path.join("bars", "geq6wrong_21017_geq6correct_1525_imagenet.npy"))
	parser.add_argument("--examples_csv", type=str, default=os.path.join("bars", "geq6wrong_21017_geq6correct_1525_imagenet_examples_ammended.csv"))
	parser.add_argument("--root_dir", type=str, default=None)
	parser.add_argument("--model_csv_name", type=str, default="efficientvit_base_0_224_classification_imagenet_1k",
					help="Model name to look up in imagenet_models.csv to select row in imagenet.npy")
	parser.add_argument("--imagenet_models_csv", type=str, default=os.path.join("bars", "imagenet_models.csv"),
					help="Path to bars/imagenet_models.csv containing model column names")
	parser.add_argument("--max_examples", type=int, default=500, help="Maximum number of examples to train on (per entire run, pre-partition)")
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
	# Time limit in seconds (default: 1 hour 45 minutes)
	parser.add_argument("--time_limit_seconds", type=int, default=6300, help="Wall-clock time budget in seconds before exiting after the current example. Default: 6300 (1h45m)")
	# Parallel arguments
	parser.add_argument("--num_workers", type=int, required=True, help="Total number of parallel workers (constant across instances)")
	parser.add_argument("--current_worker", type=int, required=True, help="Index of this worker in [0, num_workers-1]")
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	safe_model_name = args.model_name.replace('/', '_')
	model_out_dir = os.path.join(args.output_dir, safe_model_name)
	os.makedirs(model_out_dir, exist_ok=True)
	device = torch.device(args.device)

	# Configure timestamped logger in model output directory (include worker id)
	logger = logging.getLogger(f"train_grad_parallel_{safe_model_name}_w{int(args.current_worker)}")
	logger.setLevel(logging.INFO)
	logger.propagate = False
	for h in list(logger.handlers):
		logger.removeHandler(h)
	from time import strftime, localtime
	stamp = strftime("%Y%m%d_%H%M%S", localtime())
	log_path = os.path.join(model_out_dir, f"train_parallel_w{int(args.current_worker)}_{stamp}.log")
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

	# Validate worker args
	try:
		_ = _compute_worker_slice(0, int(args.num_workers), int(args.current_worker))
	except Exception:
		logger.exception("Invalid worker arguments")
		raise

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

	# Build wnid->index/name mapping from hierarchy JSON
	synset_to_idx, index_to_name, wnid_to_words = load_imagenet_hierarchy(args.hierarchy_json)

	# Build model (TIMM) and transforms for single-GPU, single-worker training
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
	# Store original model weights for reset
	original_state_dict = copy.deepcopy(model.state_dict())
	logger.info("Stored original model weights for reset")

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

	# Compute deterministic partition slice for this worker
	total_selected = len(wrong_examples_ordered)
	w_start, w_end = _compute_worker_slice(total_selected, int(args.num_workers), int(args.current_worker))
	worker_examples = wrong_examples_ordered[w_start:w_end]
	logger.info(f"Worker {int(args.current_worker)}/{int(args.num_workers)} assigned range [{w_start}, {w_end}) with {len(worker_examples)} examples")

	path_to_difficulty_rank = {path: i for i, path in enumerate(difficulty_ordered_paths)}

	# Output full list of rank indices to a single file from worker 0 only (idempotent)
	if int(args.current_worker) == 0:
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
	
	# Results manager (per-worker CSV) - exact filename per requirement
	worker_csv_name = os.path.join(model_out_dir, f"single_example_results_{int(args.current_worker)}.csv")
	results = ResultsWriter(model_out_dir, results_csv_path=worker_csv_name)
	results_csv = results.results_csv

	# Prepare directory for detailed step logs
	step_logs_dir = os.path.join(model_out_dir, "step_logs", f"worker_{int(args.current_worker)}")
	os.makedirs(step_logs_dir, exist_ok=True)

	# Configure whether CKA is enabled
	cka_enabled = (float(args.cka_layer_fraction) > 0.0)
	if not cka_enabled:
		logger.info("CKA disabled: cka_layer_fraction == 0. Skipping all CKA computations.")
	else:
		cka_manager = CKAManager(
			model=model,
			transform=train_tfms,
			device=device,
			model_out_dir=model_out_dir,
			logger=logger,
			cka_layer_fraction=float(args.cka_layer_fraction),
			seed=int(args.seed),
		)
		# Build or wait for a shared reference to avoid duplicate work across workers
		try:
			cka_manager.load_or_build_reference(
				difficulty_ordered_paths=wrong_examples_ordered,
				root_dir=args.root_dir,
				num_reference=50,
				wait_timeout_s=900,
			)
		except Exception:
			# Fallback to local build if waiting failed
			cka_manager.build_reference(
				difficulty_ordered_paths=wrong_examples_ordered,
				root_dir=args.root_dir,
				num_reference=50,
			)

	# Start wall-clock timer for auto-stop (~1h45m)
	start_time = time.time()

	# Train each example individually (sequential within this worker)
	processed_paths = set()  # Per-worker set; use JSON markers in step_logs for cross-run detection
	for local_idx, example_path in enumerate(worker_examples):
		global_order_idx = w_start + local_idx
		logger.info(f"\n=== Worker {int(args.current_worker)} Training Example {local_idx + 1}/{len(worker_examples)} (global idx {global_order_idx}): {example_path} ===")
		parent_name = os.path.basename(os.path.dirname(example_path))
		file_name = os.path.basename(example_path)
		short_id = f"{parent_name}_{file_name}" if parent_name else file_name
		short_id = short_id.replace('/', '_').replace('\\', '_').replace(':', '_')
		_rank = path_to_difficulty_rank.get(example_path, -1)
		# Build expected step log CSV path and skip if present (resume)
		step_log_file = os.path.join(step_logs_dir, f"example_{global_order_idx}_rank_{_rank:05d}_{short_id}_steps.csv")
		if os.path.exists(step_log_file):
			logger.info(f"Skipping already processed example (step log exists): {os.path.basename(step_log_file)}")
			continue
		full_path = resolve_full(example_path)
		try:
			total_steps, total_loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1, init_highest_softmax_prob, init_target_softmax_prob, steps_to_correct = trainer.train_on_example(
				full_path, se_config, reset_state, step_log_file
			)
			if cka_enabled:
				global_cka = cka_manager.after_example_trained(model, _rank, short_id)
			else:
				global_cka = ''
		except Exception:
			logger.error("Fatal error while training on example", exc_info=True)
			raise
		universal_rank = path_to_difficulty_rank[example_path] + 1
		results.append([global_order_idx, example_path, total_steps, total_loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1, universal_rank, global_cka, init_highest_softmax_prob, init_target_softmax_prob, steps_to_correct])
		logger.info(f"Example {local_idx + 1} completed: {total_steps} steps, loss sum: {total_loss_sum:.4f}, final loss: {final_loss:.8f}, weight distance: {weight_distance:.4f} (universal rank: {universal_rank})")

		# Auto-stop check: exit before starting the next example if limit exceeded
		elapsed_s = time.time() - start_time
		if elapsed_s >= int(args.time_limit_seconds):
			logger.info("Time limit reached (~1h45m). Exiting before starting next example.")
			break

	# If this is the last worker, wait for all worker CSVs, then merge and compute plots/metrics/summary
	is_last_worker = (int(args.current_worker) == int(args.num_workers) - 1)
	if is_last_worker:
		try:
			merged_csv = os.path.join(model_out_dir, "single_example_results.csv")
			# Wait until all expected worker CSVs exist (up to 24 hours)
			deadline = time.time() + 24 * 3600
			expected = {os.path.join(model_out_dir, f"single_example_results_{i}.csv"): i for i in range(int(args.num_workers))}
			while time.time() < deadline:
				existing = [p for p in expected.keys() if os.path.exists(p)]
				if len(existing) == len(expected):
					break
				logger.info(f"Waiting for worker CSVs... {len(existing)}/{len(expected)} present")
				time.sleep(30)
			# Find all worker CSVs and merge in worker-index order
			worker_csvs = []
			for p in glob.glob(os.path.join(model_out_dir, "single_example_results_*.csv")):
				try:
					basename = os.path.basename(p)
					wnum_str = basename.split("single_example_results_")[-1].split(".")[0]
					wnum = int(wnum_str)
				except Exception:
					continue
				worker_csvs.append((wnum, p))
			worker_csvs.sort(key=lambda t: t[0])
			if not worker_csvs:
				logger.warning("No worker CSVs found to merge; skipping merge and metrics")
			else:
				header_written = False
				with open(merged_csv, 'w', newline='', encoding='utf-8') as out_f:
					writer = csv.writer(out_f)
					for wnum, p in worker_csvs:
						with open(p, 'r', encoding='utf-8') as in_f:
							reader = csv.reader(in_f)
							header = next(reader, None)
							if header and (not header_written):
								writer.writerow(header)
								header_written = True
							for row in reader:
								if row:
									writer.writerow(row)
				logger.info(f"Merged {len(worker_csvs)} worker CSVs into {merged_csv}")

			# Build plots and metrics on the merged CSV
			merged_results = ResultsWriter(model_out_dir, results_csv_path=merged_csv)
			logger.info("Creating final plots from merged CSV...")
			merged_results.build_plots(epsilon=float(args.epsilon))
			try:
				merged_results.write_correlations(overwrite=True)
			except Exception:
				logger.exception("Failed writing correlations.json; continuing")
			merged_results.write_summary(epsilon=float(args.epsilon))
			summary_path = os.path.join(model_out_dir, "training_summary.json")
			logger.info(f"Training complete. Summary saved to {summary_path}")
			logger.info(f"Results CSV: {merged_csv}")
		except Exception:
			logger.exception("Failed to merge worker CSVs or compute final metrics")
	else:
		logger.info(f"Worker {int(args.current_worker)} completed. Per-worker results at {results_csv}")


if __name__ == "__main__":
	main()


