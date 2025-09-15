import os
from typing import Dict, List, Tuple

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

	def __init__(self, model: nn.Module, transform, device: torch.device) -> None:
		self.model = model
		self.transform = transform
		self.device = device
		self._pre_features: Dict[str, np.ndarray] = {}

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
		model.eval()
		for p in model.parameters():
			p.requires_grad_(False)
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


