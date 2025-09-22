import os
import csv
import glob
import time
from typing import List, Dict, Tuple

import torch

from .selection import ExampleSelector
from .results import ResultsWriter
from .cka_manager import CKAManager
from .coordination import WorkerCoordinator
from training_gradient_evaluator_single_loss_parallel.data import read_imagenet_paths
from training_gradient_evaluator_single_loss_parallel.engine.single_example_trainer import SingleExampleTrainer, SingleExampleConfig
from training_gradient_evaluator_single_loss_parallel.utils.imagenet import load_imagenet_hierarchy, read_imagenet_difficulty_order


class ParallelTrainingRunner:
	def __init__(self, args, logger) -> None:
		self.args = args
		self.logger = logger
		self.device = torch.device(args.device)
		self.model_out_dir = os.path.join(args.output_dir, args.model_name.replace('/', '_'))
		os.makedirs(self.model_out_dir, exist_ok=True)
		self.step_logs_dir = os.path.join(self.model_out_dir, "step_logs", f"worker_{int(args.current_worker)}")
		os.makedirs(self.step_logs_dir, exist_ok=True)
		self.coordinator = WorkerCoordinator(self.model_out_dir, int(args.num_workers), logger)

	def resolve_full(self, p: str) -> str:
		return os.path.join(self.args.root_dir, p) if self.args.root_dir and not os.path.isabs(p) else p

	def select_examples(self, mask_row_index: int) -> Tuple[List[str], Dict[str, int]]:
		all_paths = read_imagenet_paths(self.args.examples_csv)
		difficulty_ordered_paths = read_imagenet_difficulty_order(self.args.examples_csv)
		selector = ExampleSelector(self.model_out_dir, seed=int(self.args.seed))
		mask = __import__('numpy').load(self.args.bars_npy, allow_pickle=True)
		correct_mask = mask[mask_row_index]
		wrong_mask = __import__('numpy').array([False if i is None else (not bool(i)) for i in correct_mask], dtype=bool)
		wrong_indices = __import__('numpy').nonzero(wrong_mask)[0].tolist()
		candidate_indices = wrong_indices
		wrong_examples_ordered = selector.select_wrong_examples(candidate_indices, all_paths, difficulty_ordered_paths, self.args.root_dir, self.args.max_examples)
		path_to_rank = {p: i for i, p in enumerate(difficulty_ordered_paths)}
		return wrong_examples_ordered, path_to_rank

	def build_cka(self, model, train_tfms, image_pool: List[str]) -> CKAManager | None:
		if float(self.args.cka_layer_fraction) <= 0.0:
			self.logger.info("CKA disabled: cka_layer_fraction == 0. Skipping all CKA computations.")
			return None
		cka_manager = CKAManager(
			model=model,
			transform=train_tfms,
			device=self.device,
			model_out_dir=self.model_out_dir,
			logger=self.logger,
			cka_layer_fraction=float(self.args.cka_layer_fraction),
			seed=int(self.args.seed),
		)
		try:
			cka_manager.load_or_build_reference(
				difficulty_ordered_paths=image_pool,
				root_dir=self.args.root_dir,
				num_reference=50,
				wait_timeout_s=900,
			)
		except Exception:
			cka_manager.build_reference(
				difficulty_ordered_paths=image_pool,
				root_dir=self.args.root_dir,
				num_reference=50,
			)
		return cka_manager

	def run_worker(self, model, train_tfms, synset_to_idx: Dict[str, int], mask_row_index: int) -> None:
		# Status STARTED
		self.coordinator.update_status(int(self.args.current_worker), "STARTED")
		wrong_examples_ordered, path_to_rank = self.select_examples(mask_row_index)
		# Partition for this worker
		start, end = self._compute_worker_slice(len(wrong_examples_ordered), int(self.args.num_workers), int(self.args.current_worker))
		worker_examples = wrong_examples_ordered[start:end]
		# Results per-worker CSV
		results_csv_path = os.path.join(self.model_out_dir, f"single_example_results_{int(self.args.current_worker)}.csv")
		results = ResultsWriter(self.model_out_dir, results_csv_path=results_csv_path)
		trainer = SingleExampleTrainer(model, synset_to_idx, train_tfms, self.logger)
		se_config = SingleExampleConfig(
			lr=self.args.lr,
			weight_decay=self.args.weight_decay,
			max_steps=int(self.args.max_steps_per_example),
			epsilon=float(self.args.epsilon),
			device=self.args.device,
			use_amp=not bool(self.args.no_amp),
			gradient_clip_norm=(self.args.grad_clip_norm if self.args.grad_clip_norm and self.args.grad_clip_norm > 0 else None),
			amp_dtype=None,
		)
		reset_state = __import__('copy').deepcopy(model.state_dict())
		cka_manager = self.build_cka(model, train_tfms, wrong_examples_ordered)
		start_time = time.time()
		for local_idx, example_path in enumerate(worker_examples):
			global_idx = start + local_idx
			parent_name = os.path.basename(os.path.dirname(example_path))
			file_name = os.path.basename(example_path)
			short_id = (f"{parent_name}_{file_name}" if parent_name else file_name).replace('/', '_').replace('\\', '_').replace(':', '_')
			_rank = path_to_rank.get(example_path, -1)
			step_log_file = os.path.join(self.step_logs_dir, f"example_{global_idx}_rank_{_rank:05d}_{short_id}_steps.csv")
			if os.path.exists(step_log_file):
				self.logger.info(f"Skipping already processed example (step log exists): {os.path.basename(step_log_file)}")
				continue
			full_path = self.resolve_full(example_path)
			total_steps, total_loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1, init_highest_softmax_prob, init_target_softmax_prob, steps_to_correct = trainer.train_on_example(
				full_path, se_config, reset_state, step_log_file
			)
			global_cka = ''
			if cka_manager is not None:
				global_cka = cka_manager.after_example_trained(model, _rank, short_id)
			universal_rank = path_to_rank[example_path] + 1
			results.append([global_idx, example_path, total_steps, total_loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1, universal_rank, global_cka, init_highest_softmax_prob, init_target_softmax_prob, steps_to_correct])
			# Time limit check
			if (time.time() - start_time) >= int(self.args.time_limit_seconds):
				self.logger.info("Time limit reached. Exiting before starting next example.")
				break
		# Status DONE
		self.coordinator.update_status(int(self.args.current_worker), "DONE")
		# Merge if last worker
		if int(self.args.current_worker) == int(self.args.num_workers) - 1:
			self._merge_and_finalize()

	def _compute_worker_slice(self, n_total: int, num_workers: int, current_worker: int) -> Tuple[int, int]:
		base = n_total // num_workers
		rem = n_total % num_workers
		start = current_worker * base + min(current_worker, rem)
		end = start + base + (1 if current_worker < rem else 0)
		return start, end

	def _merge_and_finalize(self) -> None:
		self.coordinator.wait_for_all_done(timeout_s=24 * 3600)
		# Ensure all worker CSVs present
		expected_csvs = [os.path.join(self.model_out_dir, f"single_example_results_{i}.csv") for i in range(int(self.args.num_workers))]
		deadline = time.time() + 24 * 3600
		while time.time() < deadline:
			if all(os.path.exists(p) for p in expected_csvs):
				break
			time.sleep(30)
		merged_csv = os.path.join(self.model_out_dir, "single_example_results.csv")
		worker_csvs = []
		for p in glob.glob(os.path.join(self.model_out_dir, "single_example_results_*.csv")):
			try:
				wnum_str = os.path.basename(p).split("single_example_results_")[-1].split(".")[0]
				wnum = int(wnum_str)
			except Exception:
				continue
			worker_csvs.append((wnum, p))
		worker_csvs.sort(key=lambda t: t[0])
		if not worker_csvs:
			self.logger.warning("No worker CSVs found to merge; skipping merge and metrics")
			return
		with open(merged_csv, 'w', newline='', encoding='utf-8') as out_f:
			writer = csv.writer(out_f)
			header_written = False
			for _, p in worker_csvs:
				with open(p, 'r', encoding='utf-8') as in_f:
					reader = csv.reader(in_f)
					header = next(reader, None)
					if header and not header_written:
						writer.writerow(header)
						header_written = True
					for row in reader:
						if row:
							writer.writerow(row)
		self.logger.info(f"Merged {len(worker_csvs)} worker CSVs into {merged_csv}")
		from .results import ResultsWriter as _RW
		merged_results = _RW(self.model_out_dir, results_csv_path=merged_csv)
		merged_results.build_plots(epsilon=float(self.args.epsilon))
		try:
			merged_results.write_correlations(overwrite=True)
		except Exception:
			self.logger.exception("Failed writing correlations.json; continuing")
		merged_results.write_summary(epsilon=float(self.args.epsilon))
		self.logger.info(f"Training complete. Summary saved to {os.path.join(self.model_out_dir, 'training_summary.json')}")


