import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AmpContext:
	use_amp: bool
	use_scaler: bool
	amp_dtype: Optional[str]
	device_type: str

	def autocast_enabled(self) -> bool:
		return bool(self.use_amp and self.device_type == 'cuda')

	def torch_dtype(self):
		if (self.amp_dtype or 'float16') == 'bfloat16':
			return torch.bfloat16
		return torch.float16


def build_amp_context(use_amp: bool, amp_dtype: Optional[str], device: torch.device) -> AmpContext:
	use_scaler = bool(use_amp and device.type == 'cuda' and (amp_dtype or 'float16') == 'float16')
	return AmpContext(use_amp=bool(use_amp), use_scaler=use_scaler, amp_dtype=amp_dtype, device_type=device.type)


def create_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
	return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)


def set_eval_for_small_batch_modules(model: nn.Module) -> List[nn.Module]:
	_eval_modules = []
	for module in model.modules():
		if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout)):
			_eval_modules.append(module)
			module.eval()
	return _eval_modules


def restore_training(modules: List[nn.Module]) -> None:
	for m in modules:
		m.train()


def get_param_buckets(model: nn.Module) -> List[str]:
	"""Return stable parameter buckets by first name token (before first dot)."""
	buckets: List[str] = []
	seen = set()
	for name, _ in model.named_parameters():
		prefix = name.split('.')[0] if '.' in name else name
		if prefix not in seen:
			seen.add(prefix)
			buckets.append(prefix)
	return buckets


def compute_grad_mass_distribution(model: nn.Module, buckets: List[str]) -> np.ndarray:
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


def forward_backward_step(
	model: nn.Module,
	x: torch.Tensor,
	target: torch.Tensor,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	amp: AmpContext,
	scaler: torch.amp.GradScaler,
	gradient_clip_norm: Optional[float],
	param_buckets: List[str],
) -> Tuple[float, np.ndarray, int]:
	optimizer.zero_grad(set_to_none=True)
	if amp.autocast_enabled():
		with torch.amp.autocast('cuda', dtype=amp.torch_dtype()):
			logits = model(x)
			loss = criterion(logits, target)
		if amp.use_scaler:
			scaler.scale(loss).backward()
			# Capture grad mass then clip, step
			scaler.unscale_(optimizer)
			gmass = compute_grad_mass_distribution(model, param_buckets)
			if gradient_clip_norm:
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			gmass = compute_grad_mass_distribution(model, param_buckets)
			if gradient_clip_norm:
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
			optimizer.step()
	else:
		logits = model(x)
		loss = criterion(logits, target)
		loss.backward()
		gmass = compute_grad_mass_distribution(model, param_buckets)
		if gradient_clip_norm:
			torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
		optimizer.step()
	with torch.no_grad():
		logits_after = model(x)
		pred_after = int(torch.argmax(logits_after, dim=-1).item())
	return float(loss.item()), gmass, pred_after


def compute_weight_distance(initial_params: Dict[str, torch.Tensor], model: nn.Module) -> float:
	final_params: Dict[str, torch.Tensor] = {n: p.data.clone().cpu() for n, p in model.named_parameters()}
	wd = 0.0
	for n, p0 in initial_params.items():
		if n in final_params:
			d = (final_params[n].flatten().float() - p0.flatten().float())
			wd += float(torch.sum(d * d).item())
	return float(wd ** 0.5)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
	p = np.asarray(p, dtype=np.float64)
	q = np.asarray(q, dtype=np.float64)
	p = p / (p.sum() + 1e-12)
	q = q / (q.sum() + 1e-12)
	return float(0.5 * np.abs(p - q).sum())


def wasserstein_equal_bins_1d(p: np.ndarray, q: np.ndarray) -> float:
	p = np.asarray(p, dtype=np.float64)
	q = np.asarray(q, dtype=np.float64)
	p = p / (p.sum() + 1e-12)
	q = q / (q.sum() + 1e-12)
	cdf_p = np.cumsum(p)
	cdf_q = np.cumsum(q)
	return float(np.abs(cdf_p - cdf_q).sum())



