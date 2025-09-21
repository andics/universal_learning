import os
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CKAEvaluator:
	"""Compute and plot layer-wise linear CKA between two model states.

	Workflow:
	- Instantiate with a model and transform. The instance captures layer outputs for parametric modules.
	- Call build_reference on a list of image paths to record pre-training features.
	- After training, call compute_and_save(post_model, image_paths, out_png) to compute CKA grid and save.
	"""

	def __init__(self, model: nn.Module, transform, device: torch.device, layer_whitelist: Optional[Set[str]] = None) -> None:
		self.model = model
		self.transform = transform
		self.device = device
		self._pre_features: Dict[str, np.ndarray] = {}
		self._layer_whitelist: Optional[Set[str]] = (set(layer_whitelist) if layer_whitelist is not None else None)

	def _reduce_representation(self, t: torch.Tensor) -> torch.Tensor:
		if not isinstance(t, torch.Tensor):
			return None  # type: ignore
		if t.ndim >= 4:
			dims = tuple(range(2, t.ndim))
			return t.mean(dim=dims)
		if t.ndim == 3:
			return t.mean(dim=1)
		if t.ndim == 2:
			return t
		if t.ndim == 1:
			return t.unsqueeze(0)
		return t.reshape(t.shape[0], -1)

	def _collect_layer_representations(self, model: nn.Module, image_paths: List[str]) -> Dict[str, np.ndarray]:
		from PIL import Image
		prev_training = model.training
		model.eval()
		handles = []
		captured: Dict[str, torch.Tensor] = {}

		def _make_hook(key: str):
			def hook(_m, _inp, out):
				if isinstance(out, (list, tuple)) and len(out) > 0:
					out = out[0]
				if not isinstance(out, torch.Tensor):
					return
				rep = self._reduce_representation(out)
				if rep is None:
					return
				captured[key] = rep.detach().to('cpu')
			return hook

		# Register hooks on modules that have direct parameters
		for name, module in model.named_modules():
			if name == "":
				continue
			if self._layer_whitelist is not None and name not in self._layer_whitelist:
				continue
			try:
				_has_params = any(True for _ in module.parameters(recurse=False))
			except Exception:
				_has_params = False
			if not _has_params:
				continue
			handles.append(module.register_forward_hook(_make_hook(name)))

		imgs: List[torch.Tensor] = []
		for p in image_paths:
			img = Image.open(p).convert("RGB")
			imgs.append(self.transform(img))
		batch = torch.stack(imgs, dim=0).to(self.device)

		with torch.no_grad():
			_ = model(batch)

		for h in handles:
			h.remove()

		# Restore model's original training state
		if prev_training:
			model.train()

		return {k: v.float().numpy() for k, v in captured.items()}

	@staticmethod
	def _center_features(X: torch.Tensor) -> torch.Tensor:
		return X - X.mean(dim=0, keepdim=True)

	@staticmethod
	def _linear_cka(X_np: np.ndarray, Y_np: np.ndarray) -> float:
		X = torch.as_tensor(X_np, dtype=torch.float64)
		Y = torch.as_tensor(Y_np, dtype=torch.float64)
		if X.shape[0] != Y.shape[0]:
			N = min(X.shape[0], Y.shape[0])
			X = X[:N]
			Y = Y[:N]
		Xc = CKAEvaluator._center_features(X)
		Yc = CKAEvaluator._center_features(Y)
		XtY = Xc.T @ Yc
		num = float((XtY.pow(2).sum()).item())
		XtX = Xc.T @ Xc
		YtY = Yc.T @ Yc
		den = float(torch.sqrt((XtX.pow(2).sum()) * (YtY.pow(2).sum()) + 1e-20).item())
		if den <= 0:
			return 0.0
		return float(max(0.0, min(1.0, num / den)))

	def build_reference(self, image_paths: List[str]) -> None:
		self._pre_features = self._collect_layer_representations(self.model, image_paths)

	def compute_matrix(self, post_model: nn.Module, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
		post_feats = self._collect_layer_representations(post_model, image_paths)
		common = sorted(set(self._pre_features.keys()) & set(post_feats.keys()))
		L = len(common)
		M = np.zeros((L, L), dtype=np.float32)
		for i, li in enumerate(common):
			Xi = self._pre_features[li]
			for j, lj in enumerate(common):
				Yj = post_feats[lj]
				M[i, j] = CKAEvaluator._linear_cka(Xi, Yj)
		return M, common

	def compute_global_cka(self, post_model: nn.Module, image_paths: List[str]) -> float:
		"""Concatenate per-layer features across all common layers and compute a single linear CKA.
		This summarizes representational similarity across the whole network for the given batch.
		"""
		post_feats = self._collect_layer_representations(post_model, image_paths)
		common = sorted(set(self._pre_features.keys()) & set(post_feats.keys()))
		if not common:
			return float('nan')
		X_parts: List[np.ndarray] = []
		Y_parts: List[np.ndarray] = []
		# Ensure consistent sample counts; use minimum across layers if needed
		N = None
		for name in common:
			Xi = self._pre_features[name]
			Yi = post_feats[name]
			if N is None:
				N = min(Xi.shape[0], Yi.shape[0])
			else:
				N = min(int(N), Xi.shape[0], Yi.shape[0])
		if N is None or N <= 0:
			return float('nan')
		for name in common:
			Xi = self._pre_features[name][:N]
			Yi = post_feats[name][:N]
			# Align feature dims if they differ
			Di = min(Xi.shape[1], Yi.shape[1])
			X_parts.append(Xi[:, :Di])
			Y_parts.append(Yi[:, :Di])
		X_concat = np.concatenate(X_parts, axis=1)
		Y_concat = np.concatenate(Y_parts, axis=1)
		return CKAEvaluator._linear_cka(X_concat, Y_concat)

	@staticmethod
	def save_plot(M: np.ndarray, layer_names: List[str], out_path: str) -> None:
		plt.figure(figsize=(max(6, len(layer_names) * 0.3), max(5, len(layer_names) * 0.3)))
		plt.imshow(M, vmin=0.0, vmax=1.0, cmap='viridis', interpolation='nearest')
		plt.colorbar(label='Linear CKA')
		plt.xticks(ticks=range(len(layer_names)), labels=layer_names, rotation=90, fontsize=6)
		plt.yticks(ticks=range(len(layer_names)), labels=layer_names, fontsize=6)
		plt.tight_layout()
		plt.savefig(out_path, dpi=200)
		plt.close()


class CKAManager:
		"""High-level, reproducible CKA pipeline wrapper.

		Responsibilities:
		- Random, seed-stable selection of CKA layers and image batch
		- Building pre-training CKA reference
		- Saving baseline plot/JSON
		- Computing per-example CKA outputs during training
		"""

		def __init__(
			self,
			model: nn.Module,
			transform,
			device: torch.device,
			layer_fraction: float,
			seed: int,
			root_dir: Optional[str],
			difficulty_paths: List[str],
			model_out_dir: str,
			logger,
			num_images: int = 50,
			plot_interval_ranks: int = 1000,
		) -> None:
			import random as _random
			self.model = model
			self.transform = transform
			self.device = device
			self.layer_fraction = max(0.0, float(layer_fraction))
			self.seed = int(seed)
			self.root_dir = root_dir
			self.difficulty_paths = list(difficulty_paths)
			self.model_out_dir = model_out_dir
			self.logger = logger
			self.num_images = int(num_images)
			self.plot_interval_ranks = int(plot_interval_ranks)
			self.rng = _random.Random(self.seed)
			self.cka_dir = os.path.join(model_out_dir, "CKA_plots")
			self.cka_json_dir = os.path.join(model_out_dir, "CKA_jsons")
			os.makedirs(self.cka_dir, exist_ok=True)
			os.makedirs(self.cka_json_dir, exist_ok=True)
			self._layer_whitelist: Optional[Set[str]] = None
			self._cka_paths: List[str] = []
			self._evaluator: Optional[CKAEvaluator] = None
			self._last_plot_rank: Optional[int] = None

		def _resolve_full(self, p: str) -> str:
			return os.path.join(self.root_dir, p) if self.root_dir and not os.path.isabs(p) else p

		def _select_eligible_layers(self) -> List[str]:
			eligible: List[str] = []
			for name, module in self.model.named_modules():
				if name == "":
					continue
				try:
					_has_params = any(True for _ in module.parameters(recurse=False))
				except Exception:
					_has_params = False
				if not _has_params:
					continue
				eligible.append(name)
			# Sort for seed stability across Python versions
			return sorted(eligible)

		def _sample_layers(self) -> Optional[Set[str]]:
			layers = self._select_eligible_layers()
			if not layers:
				self.logger.warning("No eligible layers found for CKA hooks; defaulting to all layers")
				return None
			if self.layer_fraction <= 0.0:
				return set()  # indicates disabled
			frac = min(1.0, max(0.0, float(self.layer_fraction)))
			k = max(1, int(round(frac * len(layers))))
			try:
				chosen = set(self.rng.sample(layers, k))
			except Exception:
				chosen = set(layers[:k])
			self.logger.info(f"CKA layer subset: selecting {len(chosen)}/{len(layers)} layers (~{int(round(frac*100))}%)")
			# Persist chosen layers for reproducibility
			try:
				with open(os.path.join(self.model_out_dir, "cka_layers_chosen.txt"), 'w', encoding='utf-8') as _f:
					_f.write("\n".join(sorted(chosen)))
			except Exception:
				self.logger.exception("Failed to write cka_layers_chosen.txt", exc_info=True)
			return chosen

		def _collect_valid_images(self) -> List[str]:
			valid: List[str] = []
			for p in self.difficulty_paths:
				if isinstance(p, str) and p.strip().lower() in {"none", "null"}:
					continue
				full = self._resolve_full(p)
				if os.path.exists(full):
					valid.append(full)
			return valid

		def _sample_images(self) -> List[str]:
			valid = self._collect_valid_images()
			if not valid:
				self.logger.warning("No valid images available for CKA reference; CKA will be disabled")
				return []
			k = min(self.num_images, len(valid))
			try:
				chosen = self.rng.sample(valid, k)
			except Exception:
				chosen = valid[:k]
			if len(chosen) < self.num_images:
				self.logger.warning("Fewer than %d valid images available for CKA reference; proceeding with %d", self.num_images, len(chosen))
			# Persist chosen image paths
			try:
				with open(os.path.join(self.model_out_dir, "cka_image_paths.txt"), 'w', encoding='utf-8') as _f:
					_f.write("\n".join(chosen))
			except Exception:
				self.logger.exception("Failed to write cka_image_paths.txt", exc_info=True)
			return chosen

		def setup_baseline(self) -> None:
			"""Select layers and images, build reference, and save baseline outputs."""
			if self.layer_fraction <= 0.0:
				self._layer_whitelist = set()
				return
			self._layer_whitelist = self._sample_layers()
			if self._layer_whitelist is not None and len(self._layer_whitelist) == 0:
				# Explicitly disabled
				return
			self._cka_paths = self._sample_images()
			if not self._cka_paths:
				return
			self._evaluator = CKAEvaluator(self.model, self.transform, self.device, layer_whitelist=self._layer_whitelist)
			self._evaluator.build_reference(self._cka_paths)
			# Baseline plot/JSON (model vs itself)
			try:
				M0, layer_names0 = self._evaluator.compute_matrix(self.model, self._cka_paths)
				no_train_png = os.path.join(self.cka_dir, "CKA_no_training.png")
				CKAEvaluator.save_plot(M0, layer_names0, no_train_png)
				cka0_diag = {str(layer_names0[i]): float(M0[i, i]) for i in range(len(layer_names0))}
				no_train_json = os.path.join(self.cka_json_dir, "CKA_no_training.json")
				with open(no_train_json, 'w', encoding='utf-8') as jf:
					import json as _json
					_json.dump(cka0_diag, jf, ensure_ascii=False, indent=2)
			except Exception:
				self.logger.exception("Failed to write baseline CKA_no_training plot/JSON", exc_info=True)

		def is_enabled(self) -> bool:
			if self.layer_fraction <= 0.0:
				return False
			return bool(self._evaluator is not None and self._cka_paths)

		def after_example(self, post_model: nn.Module, example_path: str, short_id: str, rank: int) -> float | str:
			"""Compute and persist CKA artifacts after training on one example.

			Returns global CKA value (float) or empty string if disabled.
			"""
			if not self.is_enabled():
				return ''
			assert self._evaluator is not None
			try:
				M, layer_names = self._evaluator.compute_matrix(post_model, self._cka_paths)
				global_cka = self._evaluator.compute_global_cka(post_model, self._cka_paths)
				# Save per-layer diagonal JSON
				try:
					cka_diag = {str(layer_names[i]): float(M[i, i]) for i in range(len(layer_names))}
					cka_json_name = f"rank_{int(rank):05d}_{short_id}.json"
					cka_json_path = os.path.join(self.cka_json_dir, cka_json_name)
					with open(cka_json_path, 'w', encoding='utf-8') as jf:
						import json as _json
						_json.dump(cka_diag, jf, ensure_ascii=False, indent=2)
				except Exception:
					self.logger.exception("Failed to write CKA JSON for example", exc_info=True)
				# Save plot at rank intervals
				if self._last_plot_rank is None or (int(rank) - int(self._last_plot_rank)) >= self.plot_interval_ranks:
					cka_filename = f"rank_{int(rank):05d}_{short_id}.png"
					cka_out = os.path.join(self.cka_dir, cka_filename)
					CKAEvaluator.save_plot(M, layer_names, cka_out)
					self._last_plot_rank = int(rank)
				return float(global_cka)
			except Exception:
				self.logger.exception("CKA computation failed after example", exc_info=True)
				return ''

