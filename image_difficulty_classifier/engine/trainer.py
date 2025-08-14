import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..utils.checkpoint import ensure_dir, save_checkpoint, get_latest_checkpoint, load_checkpoint
from ..utils.seed import capture_rng_states, restore_rng_states
from .metrics import EpochMetrics


@dataclass
class TrainConfig:
    output_dir: str
    epochs: int = 10
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 0.01
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_every_fraction: float = 0.2
    gradient_clip_norm: Optional[float] = None


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

        ensure_dir(self.config.output_dir)
        self.checkpoints_dir = os.path.join(self.config.output_dir, "checkpoints")
        ensure_dir(self.checkpoints_dir)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.global_step = 0
        self.start_epoch = 0

        self.model.to(self.config.device)

    def maybe_resume(self, resume: bool = True) -> None:
        if not resume:
            return
        latest = get_latest_checkpoint(self.checkpoints_dir)
        if not latest:
            return
        payload = load_checkpoint(latest, map_location=self.config.device)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler and payload.get("scheduler_state"):
            self.scheduler.load_state_dict(payload["scheduler_state"])  # type: ignore[attr-defined]
        if self.scaler and payload.get("scaler_state"):
            self.scaler.load_state_dict(payload["scaler_state"])  # type: ignore[attr-defined]
        self.global_step = int(payload.get("global_step", 0))
        self.start_epoch = int(payload.get("epoch", 0))
        restore_rng_states(payload.get("rng_state"))
        if self.logger:
            self.logger.info(f"Resumed from {latest} at epoch {self.start_epoch}, step {self.global_step}")

    def _save(self, epoch: int) -> None:
        state: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "rng_state": capture_rng_states(),
            "config": asdict(self.config),
        }
        path = save_checkpoint(self.checkpoints_dir, self.global_step, state)
        if self.logger:
            self.logger.info(f"Saved checkpoint: {path}")

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        metrics = EpochMetrics()
        num_steps_per_epoch = math.ceil(len(self.train_loader.dataset) / self.config.batch_size)
        checkpoint_interval = max(1, int(num_steps_per_epoch * self.config.checkpoint_every_fraction))

        for step, (images, targets) in enumerate(self.train_loader, start=1):
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

            if step % checkpoint_interval == 0:
                self._save(epoch)

        return metrics.results()

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        metrics = EpochMetrics()
        for images, targets in loader:
            images = images.to(self.config.device, non_blocking=True)
            targets = targets.to(self.config.device, non_blocking=True)
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            metrics.update(float(loss.detach().item()), logits.detach(), targets)
        return metrics.results()

    def fit(self) -> Dict[str, float]:
        for epoch in range(self.start_epoch, self.config.epochs):
            if self.logger:
                self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            train_metrics = self._run_epoch(epoch)
            val_metrics = self._evaluate(self.val_loader)
            if self.logger:
                self.logger.info(f"Train: {train_metrics} | Val: {val_metrics}")
            self._save(epoch)
        test_metrics = self._evaluate(self.test_loader)
        if self.logger:
            self.logger.info(f"Test: {test_metrics}")
        return test_metrics


