import math
import time
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ..engine.metrics import EpochMetrics


@dataclass
class TrainConfig:
	output_dir: str
	epochs: int = 10
	batch_size: int = 128
	lr: float = 5e-4
	weight_decay: float = 0.01
	num_workers: int = 4
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	checkpoint_every_fraction: float = 0.25
	gradient_clip_norm: Optional[float] = None
	log_image_paths: bool = True


class Trainer:
	def __init__(
		self,
		model: torch.nn.Module,
		train_loader: DataLoader,
		val_loader: DataLoader,
		test_loader: DataLoader,
		optimizer: torch.optim.Optimizer,
		scheduler: Optional[Any],
		config: TrainConfig,
		scaler: Optional[torch.cuda.amp.GradScaler] = None,
		logger=None,
		train_image_paths: Optional[Sequence[str]] = None,
	) -> None:
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.config = config
		self.scaler = scaler
		self.logger = logger
		self.train_image_paths = train_image_paths

		os.makedirs(self.config.output_dir, exist_ok=True)
		self.criterion = torch.nn.CrossEntropyLoss()
		self.global_step = 0
		self.start_epoch = 0

		self.model.to(self.config.device)

	def _run_epoch(self, epoch: int) -> Dict[str, float]:
		self.model.train()
		metrics = EpochMetrics()
		num_steps_per_epoch = max(1, math.ceil(len(self.train_loader.dataset) / self.config.batch_size))
		checkpoint_interval = max(1, int(num_steps_per_epoch * self.config.checkpoint_every_fraction))
		epoch_start_time = time.time()

		for step, batch in enumerate(self.train_loader, start=1):
			# Batch may be (images, targets, paths) from our dataset
			if len(batch) == 3:
				images, targets, batch_paths = batch
			else:
				images, targets = batch
				batch_paths = None
			step_start_time = time.time()
			images = images.to(self.config.device, non_blocking=True)
			targets = targets.to(self.config.device, non_blocking=True)

			self.optimizer.zero_grad(set_to_none=True)
			if self.scaler:
				with torch.autocast(device_type=self.config.device.split(":")[0], dtype=torch.float16):
					logits = self.model(images)
					loss = self.criterion(logits, targets)
				self.scaler.scale(loss).backward()
				if self.config.gradient_clip_norm:
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
				self.scaler.step(self.optimizer)
				self.scaler.update()
			else:
				logits = self.model(images)
				loss = self.criterion(logits, targets)
				loss.backward()
				if self.config.gradient_clip_norm:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
				self.optimizer.step()

			if self.scheduler:
				self.scheduler.step()

			metrics.update(float(loss.detach().item()), logits.detach(), targets)
			self.global_step += 1

			# Per-step logging (verbose)
			if self.logger:
				with torch.no_grad():
					preds = torch.argmax(logits, dim=1)
					correct_mask = (preds == targets)
					batch_acc = float(correct_mask.sum().item()) / max(int(targets.numel()), 1)
					if batch_paths is not None and self.config.log_image_paths:
						right_paths = [p for p, c in zip(batch_paths, correct_mask.cpu().tolist()) if c]
						wrong_paths = [p for p, c in zip(batch_paths, correct_mask.cpu().tolist()) if not c]
					else:
						right_paths = []
						wrong_paths = []
				lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
				step_time = time.time() - step_start_time
				imgs_per_sec = int(images.size(0) / max(step_time, 1e-8))
				self.logger.info(
					f"epoch={epoch+1} step={step}/{num_steps_per_epoch} global_step={self.global_step} "
					f"loss={float(loss.detach().item()):.4f} batch_acc={batch_acc:.4f} lr={lr:.6g} "
					f"time={step_time:.3f}s ips={imgs_per_sec}/s"
				)
				if batch_paths is not None and self.config.log_image_paths:
					if right_paths:
						self.logger.info("RIGHT: " + " | ".join(right_paths))
					if wrong_paths:
						self.logger.info("WRONG: " + " | ".join(wrong_paths))

			if step % checkpoint_interval == 0:
				# lightweight periodic checkpoint: state_dict only
				ckpt_path = os.path.join(self.config.output_dir, f"ckpt_step_{self.global_step}.pt")
				torch.save({"model": self.model.state_dict(), "global_step": self.global_step, "epoch": epoch}, ckpt_path)
				if self.logger:
					self.logger.info(f"Saved checkpoint: {ckpt_path}")

		return metrics.results()

	@torch.no_grad()
	def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
		self.model.eval()
		metrics = EpochMetrics()
		start_time = time.time()
		for batch in loader:
			if len(batch) == 3:
				images, targets, _ = batch
			else:
				images, targets = batch
			images = images.to(self.config.device, non_blocking=True)
			targets = targets.to(self.config.device, non_blocking=True)
			logits = self.model(images)
			loss = self.criterion(logits, targets)
			metrics.update(float(loss.detach().item()), logits.detach(), targets)
		results = metrics.results()
		results["time_sec"] = float(time.time() - start_time)
		return results

	def fit(self) -> Dict[str, float]:
		for epoch in range(self.start_epoch, self.config.epochs):
			if self.logger:
				self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
			train_start = time.time()
			train_metrics = self._run_epoch(epoch)
			train_metrics["time_sec"] = float(time.time() - train_start)
			val_metrics = self._evaluate(self.val_loader)
			if self.logger:
				self.logger.info(
					f"Train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
					f"samples={int(train_metrics['num_samples'])} steps={int(train_metrics['num_steps'])} "
					f"time={train_metrics['time_sec']:.2f}s | "
					f"Val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
					f"samples={int(val_metrics['num_samples'])} time={val_metrics['time_sec']:.2f}s"
				)
		# final test
		test_metrics = self._evaluate(self.test_loader)
		if self.logger:
			self.logger.info(
				f"Test: loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} "
				f"samples={int(test_metrics['num_samples'])} time={test_metrics['time_sec']:.2f}s"
			)
		return test_metrics


