import os
import json
from typing import List, Optional, Set

import numpy as np
import torch
import time

from .cka import CKAEvaluator


class CKAManager:
	"""High-level manager that encapsulates all CKA-related logic.

	Responsibilities:
	- Choose a random subset of parametric layers based on a fraction
	- Uniformly sample reference images across a difficulty-ordered list
	- Build and store CKA reference features
	- Compute per-example CKA results and save plots/JSONs
	"""

	def __init__(
		self,
		model,
		transform,
		device: torch.device,
		model_out_dir: str,
		logger,
		cka_layer_fraction: float = 0.05,
		seed: Optional[int] = None,
	) -> None:
		self.model = model
		self.transform = transform
		self.device = device
		self.model_out_dir = model_out_dir
		self.logger = logger
		self.cka_layer_fraction = float(cka_layer_fraction)
		self.seed = int(seed) if seed is not None else None

		self.cka_dir = os.path.join(self.model_out_dir, "CKA_plots")
		self.cka_json_dir = os.path.join(self.model_out_dir, "CKA_jsons")
		self.cka_ref_npz = os.path.join(self.model_out_dir, "CKA_reference_features.npz")
		self.cka_image_paths_txt = os.path.join(self.model_out_dir, "cka_image_paths.txt")
		self.cka_ref_lock = os.path.join(self.model_out_dir, "CKA_reference.lock")
		os.makedirs(self.cka_dir, exist_ok=True)
		os.makedirs(self.cka_json_dir, exist_ok=True)

		self._layer_whitelist: Optional[Set[str]] = None
		self._cka_paths: List[str] = []
		self._last_cka_plot_rank: Optional[int] = None
		self._evaluator: Optional[CKAEvaluator] = None

	def _choose_random_layers(self) -> Optional[Set[str]]:
		"""Select a random subset of parametric layer names.

		If fraction is <=0, returns None (use all eligible layers). If >1, clamps to 1.
		"""
		eligible_layers: List[str] = []
		for lname, module in self.model.named_modules():
			if lname == "":
				continue
			try:
				_has_params = any(True for _ in module.parameters(recurse=False))
			except Exception:
				_has_params = False
			if not _has_params:
				continue
			eligible_layers.append(lname)

		num_layers = len(eligible_layers)
		if num_layers <= 0:
			self.logger.warning("No eligible layers found for CKA hooks; defaulting to all layers")
			return None

		frac = max(0.0, min(1.0, float(self.cka_layer_fraction)))
		if frac <= 0.0:
			return None
		k = max(1, int(round(frac * num_layers)))

		from random import Random
		rng = Random(self.seed)
		try:
			chosen = set(rng.sample(eligible_layers, k))
		except Exception:
			chosen = set(eligible_layers[:k])

		self.logger.info(f"CKA layer subset: selecting {len(chosen)}/{num_layers} layers (~{int(round(frac*100))}%)")
		try:
			with open(os.path.join(self.model_out_dir, "cka_layers_chosen.txt"), 'w', encoding='utf-8') as f:
				f.write("\n".join(sorted(chosen)))
		except Exception:
			self.logger.exception("Failed to write cka_layers_chosen.txt", exc_info=True)
		return chosen

	def _read_chosen_layers(self) -> Optional[Set[str]]:
		"""If a chosen layer list exists on disk, load and use it for consistency across workers."""
		path = os.path.join(self.model_out_dir, "cka_layers_chosen.txt")
		if not os.path.exists(path):
			return None
		try:
			with open(path, 'r', encoding='utf-8') as f:
				layers = [line.strip() for line in f if line.strip()]
			self._layer_whitelist = set(layers)
			return self._layer_whitelist
		except Exception:
			return None

	def _save_reference_npz(self) -> None:
		"""Persist pre-training features so workers can load instead of recomputing."""
		if self._evaluator is None:
			return
		try:
			feat = getattr(self._evaluator, "_pre_features", {}) or {}
			if not isinstance(feat, dict) or not feat:
				return
			# Save as object arrays; allow_pickle needed on load
			np.savez(self.cka_ref_npz, **{k: v for k, v in feat.items()})
		except Exception:
			self.logger.exception("Failed to save CKA reference features", exc_info=True)

	def _load_reference_npz(self) -> bool:
		"""Load pre-training features if present."""
		if not os.path.exists(self.cka_ref_npz):
			return False
		try:
			self._evaluator = CKAEvaluator(self.model, self.transform, self.device, layer_whitelist=self._layer_whitelist)
			data = np.load(self.cka_ref_npz, allow_pickle=True)
			feat = {k: data[k] for k in data.files}
			# Convert to numpy arrays (already), assign directly
			setattr(self._evaluator, "_pre_features", feat)
			# Populate CKA image path list from persisted file so other workers can compute post features
			if os.path.exists(self.cka_image_paths_txt):
				try:
					with open(self.cka_image_paths_txt, 'r', encoding='utf-8') as f:
						paths = [line.strip() for line in f if line.strip()]
					self._cka_paths = paths
				except Exception:
					self._cka_paths = []
			return True
		except Exception:
			self.logger.exception("Failed to load CKA reference features", exc_info=True)
			return False

	@staticmethod
	def _resolve_full(p: str, root_dir: Optional[str]) -> str:
		if root_dir and not os.path.isabs(p):
			return os.path.join(root_dir, p)
		return p

	@staticmethod
	def _uniform_sample_indices(total: int, k: int) -> List[int]:
		if total <= 0 or k <= 0:
			return []
		if total <= k:
			return list(range(total))
		idxs = np.linspace(0, total - 1, num=k)
		idxs = np.unique(np.round(idxs).astype(int))
		# If rounding collapsed some, pad by random unique choices to reach k
		if idxs.size < k:
			remaining = [i for i in range(total) if i not in idxs]
			from random import Random
			rng = Random(0)
			pad = rng.sample(remaining, k - int(idxs.size))
			idxs = np.concatenate([idxs, np.array(sorted(pad), dtype=int)])
		return list(map(int, idxs))

	def build_reference(
		self,
		difficulty_ordered_paths: List[str],
		root_dir: Optional[str],
		num_reference: int = 50,
	) -> None:
		"""Select uniformly spaced valid images and build the CKA reference features.
		Also selects the random layer subset and writes the baseline plot/JSON.
		"""
		# Choose random layers once
		# Prefer persisted chosen layers (for cross-worker consistency); otherwise choose and persist
		self._layer_whitelist = self._read_chosen_layers() or self._choose_random_layers()

		# Resolve existing files in order
		valid_full_paths: List[str] = []
		for p in difficulty_ordered_paths:
			if isinstance(p, str) and p.strip().lower() in {"none", "null"}:
				continue
			full = self._resolve_full(p, root_dir)
			if os.path.exists(full):
				valid_full_paths.append(full)

		if len(valid_full_paths) == 0:
			self.logger.warning("No valid images available for CKA reference; CKA will be skipped")
			self._cka_paths = []
			self._evaluator = CKAEvaluator(self.model, self.transform, self.device, layer_whitelist=self._layer_whitelist)
			return

		indices = self._uniform_sample_indices(len(valid_full_paths), int(num_reference))
		self._cka_paths = [valid_full_paths[i] for i in indices]
		if len(self._cka_paths) < int(num_reference):
			self.logger.warning("Fewer than %d valid images available for CKA reference; proceeding with %d", int(num_reference), len(self._cka_paths))

		self._evaluator = CKAEvaluator(self.model, self.transform, self.device, layer_whitelist=self._layer_whitelist)
		self._evaluator.build_reference(self._cka_paths)
		# Persist reference features for other workers
		self._save_reference_npz()
		# Persist image paths so other workers can reuse the same batch
		try:
			with open(self.cka_image_paths_txt, 'w', encoding='utf-8') as f:
				f.write("\n".join(self._cka_paths))
		except Exception:
			self.logger.exception("Failed to write cka_image_paths.txt", exc_info=True)

		# Baseline (model vs itself before any training) — avoid concurrent re-writes
		try:
			no_train_png = os.path.join(self.cka_dir, "CKA_no_training.png")
			no_train_json = os.path.join(self.cka_json_dir, "CKA_no_training.json")
			if not (os.path.exists(no_train_png) and os.path.exists(no_train_json)):
				M0, layer_names0 = self._evaluator.compute_matrix(self.model, self._cka_paths)
				CKAEvaluator.save_plot(M0, layer_names0, no_train_png)
				cka0_diag = {str(layer_names0[i]): float(M0[i, i]) for i in range(len(layer_names0))}
				with open(no_train_json, 'w', encoding='utf-8') as jf:
					json.dump(cka0_diag, jf, ensure_ascii=False, indent=2)
		except Exception:
			self.logger.exception("Failed to write baseline CKA_no_training plot/JSON", exc_info=True)

	def load_or_build_reference(
		self,
		difficulty_ordered_paths: List[str],
		root_dir: Optional[str],
		num_reference: int = 50,
		wait_timeout_s: int = 900,
	) -> None:
		"""Load reference features if available; otherwise build while other workers wait.

		Uses a simple file lock to ensure only one builder across processes.
		"""
		# Ensure layer whitelist known
		self._layer_whitelist = self._read_chosen_layers() or self._layer_whitelist
		if self._load_reference_npz():
			return
		# Acquire builder lock if possible
		lock_acquired = False
		try:
			fd = os.open(self.cka_ref_lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
			os.close(fd)
			lock_acquired = True
		except Exception:
			lock_acquired = False
		if lock_acquired:
			try:
				self.build_reference(difficulty_ordered_paths, root_dir, num_reference=num_reference)
			finally:
				try:
					os.remove(self.cka_ref_lock)
				except Exception:
					pass
			return
		# Wait for reference to appear
		start = time.time()
		while (time.time() - start) < float(wait_timeout_s):
			if self._load_reference_npz():
				return
			time.sleep(1.0)
		# Timeout: as fallback, build locally (last resort)
		self.build_reference(difficulty_ordered_paths, root_dir, num_reference=num_reference)

	def after_example_trained(self, model, example_rank: int, short_id: str) -> str:
		"""Compute per-example CKA, save JSONs, and optionally plots. Returns global CKA as string."""
		if self._evaluator is None or not self._cka_paths:
			return ''
		try:
			M, layer_names = self._evaluator.compute_matrix(model, self._cka_paths)
			global_cka_val = self._evaluator.compute_global_cka(model, self._cka_paths)
			# Per-layer diagonal JSON
			cka_diag = {str(layer_names[i]): float(M[i, i]) for i in range(len(layer_names))}
			cka_json_name = f"rank_{int(example_rank):05d}_{short_id}.json"
			cka_json_path = os.path.join(self.cka_json_dir, cka_json_name)
			with open(cka_json_path, 'w', encoding='utf-8') as jf:
				json.dump(cka_diag, jf, ensure_ascii=False, indent=2)
			# Plot at coarse intervals (>=1000 ranks apart)
			if self._last_cka_plot_rank is None or (int(example_rank) - int(self._last_cka_plot_rank)) >= 1000:
				cka_filename = f"rank_{int(example_rank):05d}_{short_id}.png"
				cka_out = os.path.join(self.cka_dir, cka_filename)
				CKAEvaluator.save_plot(M, layer_names, cka_out)
				self._last_cka_plot_rank = int(example_rank)
			return f"{float(global_cka_val):.6f}"
		except Exception:
			self.logger.exception("Failed CKA computation for example", exc_info=True)
			return ''



