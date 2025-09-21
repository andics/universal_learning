import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from training_gradient_evaluator_single_loss.engine.train_utils import (
	AmpContext,
	build_amp_context,
	create_optimizer,
	set_eval_for_small_batch_modules,
	restore_training,
	get_param_buckets,
	compute_grad_mass_distribution,
	forward_backward_step,
	compute_weight_distance,
	total_variation_distance,
	wasserstein_equal_bins_1d,
)




@dataclass
class SingleExampleConfig:
	lr: float = 1e-3
	weight_decay: float = 0.0
	max_steps: int = 10000
	epsilon: float = 1e-3
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	use_amp: bool = False
	gradient_clip_norm: Optional[float] = None
	amp_dtype: Optional[str] = None  # one of {"float16", "bfloat16", None}


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
	) -> Tuple[int, float, float, float, float, float, float, float, int]:
		"""Run SGD on a single image until loss <= epsilon or max_steps.
		Returns (total_steps, loss_sum, final_loss, weight_distance, softmax_w1, grad_mass_w1, init_highest_softmax_prob, init_target_softmax_prob, steps_to_correct).
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
			# New initial metrics
			init_highest_softmax_prob = float(np.max(probs0))
			init_target_softmax_prob = float(probs0[int(target.item())])
			initial_pred = int(torch.argmax(logits0, dim=-1).item())

		self.model.train()
		# Put BN and Dropout in eval to avoid batch size=1 stat updates and stochasticity
		_eval_modules = set_eval_for_small_batch_modules(self.model)

		optimizer = create_optimizer(self.model, lr=config.lr, weight_decay=config.weight_decay)
		# AMP setup
		amp_ctx: AmpContext = build_amp_context(config.use_amp, config.amp_dtype, device)
		scaler = torch.amp.GradScaler('cuda', enabled=amp_ctx.use_scaler)
		criterion = nn.CrossEntropyLoss()
		param_buckets = get_param_buckets(self.model)

		total_loss_sum = 0.0
		first_grad_mass = None
		last_grad_mass = None
		steps_to_correct = 0 if initial_pred == int(target.item()) else -1

		# Step logging
		os.makedirs(os.path.dirname(step_log_csv_path), exist_ok=True)
		import csv
		with open(step_log_csv_path, "w", newline="", encoding="utf-8") as step_f:
			step_writer = csv.writer(step_f)
			step_writer.writerow(["step", "loss", "cumulative_loss_sum"])

			for step in range(1, int(config.max_steps) + 1):
				loss_value, gmass, pred_after = forward_backward_step(
					model=self.model,
					x=x,
					target=target,
					criterion=criterion,
					optimizer=optimizer,
					amp=amp_ctx,
					scaler=scaler,
					gradient_clip_norm=config.gradient_clip_norm,
					param_buckets=param_buckets,
				)

				# Track steps_to_correct: first step when prediction equals target
				if steps_to_correct < 0 and pred_after == int(target.item()):
					steps_to_correct = int(step)

				current_loss = float(loss_value)
				total_loss_sum += current_loss
				step_writer.writerow([step, current_loss, total_loss_sum])

				if torch.isnan(torch.tensor(current_loss)):
					# Weight distance (use parameter tensors only)
					wd = compute_weight_distance(initial_param_weights, self.model)
					with torch.no_grad():
						self.model.eval()
						logitsT = self.model(x)
						probsT = torch.softmax(logitsT, dim=-1).squeeze(0).detach().cpu().numpy()
					softmax_w1 = total_variation_distance(probs0, probsT)
					grad_mass_w1 = wasserstein_equal_bins_1d(first_grad_mass if first_grad_mass is not None else np.array([1.0]), last_grad_mass if last_grad_mass is not None else np.array([1.0]))
					restore_training(_eval_modules)
					if self.logger:
						self.logger.warning(f"NaN loss detected at step {step}. Stopping training for this example.")
					return -1, total_loss_sum, float('nan'), wd, softmax_w1, grad_mass_w1, init_highest_softmax_prob, init_target_softmax_prob, int(steps_to_correct)

				if current_loss <= float(config.epsilon):
					wd = compute_weight_distance(initial_param_weights, self.model)
					with torch.no_grad():
						self.model.eval()
						logitsT = self.model(x)
						probsT = torch.softmax(logitsT, dim=-1).squeeze(0).detach().cpu().numpy()
					softmax_w1 = total_variation_distance(probs0, probsT)
					grad_mass_w1 = wasserstein_equal_bins_1d(first_grad_mass if first_grad_mass is not None else np.array([1.0]), last_grad_mass if last_grad_mass is not None else np.array([1.0]))
					restore_training(_eval_modules)
					if self.logger:
						self.logger.info(
							f"Example {example_path} reached epsilon at step {step}, loss sum: {total_loss_sum:.4f}, final loss: {current_loss:.8f}, weight distance: {wd:.4f}"
						)
					return step, total_loss_sum, current_loss, wd, softmax_w1, grad_mass_w1, init_highest_softmax_prob, init_target_softmax_prob, int(steps_to_correct)

		# Finalize without reaching epsilon
		wd = compute_weight_distance(initial_param_weights, self.model)
		with torch.no_grad():
			self.model.eval()
			logitsT = self.model(x)
			probsT = torch.softmax(logitsT, dim=-1).squeeze(0).detach().cpu().numpy()
		softmax_w1 = total_variation_distance(probs0, probsT)
		grad_mass_w1 = wasserstein_equal_bins_1d(first_grad_mass if first_grad_mass is not None else np.array([1.0]), last_grad_mass if last_grad_mass is not None else np.array([1.0]))
		restore_training(_eval_modules)
		if self.logger:
			self.logger.info(
				f"Example {example_path} never reached epsilon after {config.max_steps} steps, loss sum: {total_loss_sum:.4f}, final loss: {current_loss:.8f}, weight distance: {wd:.4f}"
			)
		return -1, total_loss_sum, current_loss, wd, softmax_w1, grad_mass_w1, init_highest_softmax_prob, init_target_softmax_prob, int(steps_to_correct)


