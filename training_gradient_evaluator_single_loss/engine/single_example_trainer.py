import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


def _total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
	"""TV distance between categorical distributions p and q.
	This equals the 1-Wasserstein distance under a 0-1 ground metric."""
	p = np.asarray(p, dtype=np.float64)
	q = np.asarray(q, dtype=np.float64)
	p = p / (p.sum() + 1e-12)
	q = q / (q.sum() + 1e-12)
	return float(0.5 * np.abs(p - q).sum())


def _wasserstein_equal_bins_1d(p: np.ndarray, q: np.ndarray) -> float:
	"""Compute 1D W1 between discrete distributions on equally spaced bins (width=1)."""
	p = np.asarray(p, dtype=np.float64)
	q = np.asarray(q, dtype=np.float64)
	p = p / (p.sum() + 1e-12)
	q = q / (q.sum() + 1e-12)
	cdf_p = np.cumsum(p)
	cdf_q = np.cumsum(q)
	return float(np.abs(cdf_p - cdf_q).sum())


def _get_param_buckets(model: nn.Module) -> list[str]:
	"""Return stable parameter buckets by first name token (before first dot)."""
	buckets: list[str] = []
	seen = set()
	for name, _ in model.named_parameters():
		prefix = name.split('.')[0] if '.' in name else name
		if prefix not in seen:
			seen.add(prefix)
			buckets.append(prefix)
	return buckets


def _compute_grad_mass_distribution(model: nn.Module, buckets: list[str]) -> np.ndarray:
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


@dataclass
class SingleExampleConfig:
	lr: float = 1e-3
	weight_decay: float = 0.0
	max_steps: int = 10000
	epsilon: float = 1e-3
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	use_amp: bool = False
	gradient_clip_norm: Optional[float] = None


class SingleExampleTrainer:
	def __init__(self, model: nn.Module, synset_to_idx: Dict[str, int], train_tfms, logger=None) -> None:
		self.model = model
		self.synset_to_idx = synset_to_idx
		self.train_tfms = train_tfms
		self.logger = logger

	def train_on_example(
		self,
		example_path: str,
		config: SingleExampleConfig,
		reset_state_dict: Dict[str, torch.Tensor],
		step_log_csv_path: str,
	) -> Tuple[int, float, float, float, float, float]:
		"""Run SGD on a single image until loss <= epsilon or max_steps.
		Returns (total_steps, loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1).
		"""
		from PIL import Image
		from training_gradient_evaluator_single_loss.data import extract_synset_from_path

		device = torch.device(config.device)
		# Reset weights using full state_dict (includes buffers)
		self.model.load_state_dict(reset_state_dict)
		self.model.to(device)
		# Capture initial parameter values for weight distance
		initial_param_weights = {name: p.data.clone().cpu() for name, p in self.model.named_parameters()}

		# Prepare data
		image = Image.open(example_path).convert("RGB")
		x = self.train_tfms(image).unsqueeze(0).to(device)
		wnid = extract_synset_from_path(example_path)
		if wnid is None or wnid not in self.synset_to_idx:
			raise RuntimeError(f"Could not determine synset/class for path: {example_path}")
		target = torch.tensor([self.synset_to_idx[wnid]], device=device)

		# Before-training softmax
		with torch.no_grad():
			self.model.eval()
			logits0 = self.model(x)
			probs0 = torch.softmax(logits0, dim=-1).squeeze(0).detach().cpu().numpy()

		self.model.train()
		# Put BN and Dropout in eval to avoid batch size=1 stat updates and stochasticity
		_eval_modules = []
		for module in self.model.modules():
			if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout)):
				_eval_modules.append(module)
				module.eval()

		optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
		scaler = torch.amp.GradScaler('cuda', enabled=(config.use_amp and device.type == "cuda"))
		criterion = nn.CrossEntropyLoss()
		param_buckets = _get_param_buckets(self.model)

		total_loss_sum = 0.0
		first_grad_mass = None
		last_grad_mass = None

		# Step logging
		os.makedirs(os.path.dirname(step_log_csv_path), exist_ok=True)
		import csv
		with open(step_log_csv_path, "w", newline="", encoding="utf-8") as step_f:
			step_writer = csv.writer(step_f)
			step_writer.writerow(["step", "loss", "cumulative_loss_sum"])

			for step in range(1, int(config.max_steps) + 1):
				optimizer.zero_grad(set_to_none=True)
				if scaler.is_enabled():
					with torch.amp.autocast('cuda'):
						logits = self.model(x)
						loss = criterion(logits, target)
					scaler.scale(loss).backward()
					# Capture grad mass then clip, step
					scaler.unscale_(optimizer)
					gmass = _compute_grad_mass_distribution(self.model, param_buckets)
					if first_grad_mass is None:
						first_grad_mass = gmass
					last_grad_mass = gmass
					if config.gradient_clip_norm:
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.gradient_clip_norm)
					scaler.step(optimizer)
					scaler.update()
				else:
					logits = self.model(x)
					loss = criterion(logits, target)
					loss.backward()
					gmass = _compute_grad_mass_distribution(self.model, param_buckets)
					if first_grad_mass is None:
						first_grad_mass = gmass
					last_grad_mass = gmass
					if config.gradient_clip_norm:
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.gradient_clip_norm)
					optimizer.step()

				current_loss = float(loss.item())
				total_loss_sum += current_loss
				step_writer.writerow([step, current_loss, total_loss_sum])

				if torch.isnan(loss):
					# Weight distance (use parameter tensors only)
					final_params = {n: p.data.clone().cpu() for n, p in self.model.named_parameters()}
					wd = 0.0
					for n, p0 in initial_param_weights.items():
						if n in final_params:
							d = (final_params[n].flatten().float() - p0.flatten().float())
							wd += float(torch.sum(d * d).item())
					wd = float(wd ** 0.5)
					with torch.no_grad():
						self.model.eval()
						logitsT = self.model(x)
						probsT = torch.softmax(logitsT, dim=-1).squeeze(0).detach().cpu().numpy()
					softmax_w1 = _total_variation_distance(probs0, probsT)
					grad_mass_w1 = _wasserstein_equal_bins_1d(first_grad_mass if first_grad_mass is not None else np.array([1.0]), last_grad_mass if last_grad_mass is not None else np.array([1.0]))
					for m in _eval_modules:
						m.train()
					if self.logger:
						self.logger.warning(f"NaN loss detected at step {step}. Stopping training for this example.")
					return -1, total_loss_sum, float('nan'), wd, softmax_w1, grad_mass_w1

				if current_loss <= float(config.epsilon):
					final_params = {n: p.data.clone().cpu() for n, p in self.model.named_parameters()}
					wd = 0.0
					for n, p0 in initial_param_weights.items():
						if n in final_params:
							d = (final_params[n].flatten().float() - p0.flatten().float())
							wd += float(torch.sum(d * d).item())
					wd = float(wd ** 0.5)
					with torch.no_grad():
						self.model.eval()
						logitsT = self.model(x)
						probsT = torch.softmax(logitsT, dim=-1).squeeze(0).detach().cpu().numpy()
					softmax_w1 = _total_variation_distance(probs0, probsT)
					grad_mass_w1 = _wasserstein_equal_bins_1d(first_grad_mass if first_grad_mass is not None else np.array([1.0]), last_grad_mass if last_grad_mass is not None else np.array([1.0]))
					for m in _eval_modules:
						m.train()
					if self.logger:
						self.logger.info(
							f"Example {example_path} reached epsilon at step {step}, loss sum: {total_loss_sum:.4f}, final loss: {current_loss:.8f}, weight distance: {wd:.4f}"
						)
					return step, total_loss_sum, current_loss, wd, softmax_w1, grad_mass_w1

		# Finalize without reaching epsilon
		final_params = {n: p.data.clone().cpu() for n, p in self.model.named_parameters()}
		wd = 0.0
		for n, p0 in initial_param_weights.items():
			if n in final_params:
				d = (final_params[n].flatten().float() - p0.flatten().float())
				wd += float(torch.sum(d * d).item())
		wd = float(wd ** 0.5)
		with torch.no_grad():
			self.model.eval()
			logitsT = self.model(x)
			probsT = torch.softmax(logitsT, dim=-1).squeeze(0).detach().cpu().numpy()
		softmax_w1 = _total_variation_distance(probs0, probsT)
		grad_mass_w1 = _wasserstein_equal_bins_1d(first_grad_mass if first_grad_mass is not None else np.array([1.0]), last_grad_mass if last_grad_mass is not None else np.array([1.0]))
		for m in _eval_modules:
			m.train()
		if self.logger:
			self.logger.info(
				f"Example {example_path} never reached epsilon after {config.max_steps} steps, loss sum: {total_loss_sum:.4f}, final loss: {current_loss:.8f}, weight distance: {wd:.4f}"
			)
		return -1, total_loss_sum, current_loss, wd, softmax_w1, grad_mass_w1


