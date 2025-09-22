import argparse
import os, sys
from typing import Tuple
from pathlib import Path

# Ensure working directory and sys.path point to the Programming root so package imports resolve
try:
	path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
	if path_main not in sys.path:
		sys.path.append(path_main)
	os.chdir(path_main)
except Exception as _e:
	print(f"[WARN] Failed to set working dir/sys.path to Programming root: {_e}", file=sys.stderr)

import logging
import torch
import numpy as np
from training_gradient_evaluator_single_loss_parallel.utils.imagenet import default_hierarchy_json_path as _default_hierarchy_json_path, load_imagenet_hierarchy
from training_gradient_evaluator_single_loss_parallel.engine.runner import ParallelTrainingRunner



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

	# Also log resolved training configuration (values are directly in args)
	logger.info("Training configuration:")
	logger.info(f"  lr = {args.lr}")
	logger.info(f"  weight_decay = {args.weight_decay}")
	logger.info(f"  max_steps = {args.max_steps_per_example}")
	logger.info(f"  epsilon = {args.epsilon}")
	logger.info(f"  device = {args.device}")
	logger.info(f"  use_amp = {not bool(args.no_amp)}")
	logger.info(f"  amp_dtype = {args.amp_dtype}")
	logger.info(f"  gradient_clip_norm = {args.grad_clip_norm}")

	# Run
	from training_gradient_evaluator_single_loss_parallel.utils.imagenet import resolve_model_index as _resolve_idx
	mask_row_idx = _resolve_idx(args.imagenet_models_csv, args.model_csv_name)
	runner = ParallelTrainingRunner(args, logger)
	runner.run_worker(model, train_tfms, synset_to_idx, mask_row_idx)


if __name__ == "__main__":
	main()


