import math
import time
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
        try:
            restore_rng_states(payload.get("rng_state"))
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Skipping RNG state restore due to error: {e}")
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
        path = save_checkpoint(self.checkpoints_dir, self.global_step, state, epoch=epoch)
        if self.logger:
            self.logger.info(f"Saved checkpoint: {path}")

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        metrics = EpochMetrics()
        num_steps_per_epoch = math.ceil(len(self.train_loader.dataset) / self.config.batch_size)
        checkpoint_interval = max(1, int(num_steps_per_epoch * self.config.checkpoint_every_fraction))
        epoch_start_time = time.time()

        for step, (images, targets) in enumerate(self.train_loader, start=1):
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

            # Per-step logging
            if self.logger:
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    batch_acc = float((preds == targets).sum().item()) / max(int(targets.numel()), 1)
                lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
                step_time = time.time() - step_start_time
                imgs_per_sec = int(images.size(0) / max(step_time, 1e-8))
                self.logger.info(
                    f"epoch={epoch+1} step={step}/{num_steps_per_epoch} global_step={self.global_step} "
                    f"loss={float(loss.detach().item()):.4f} batch_acc={batch_acc:.4f} lr={lr:.6g} "
                    f"time={step_time:.3f}s ips={imgs_per_sec}/s"
                )

            if step % checkpoint_interval == 0:
                self._save(epoch)

        return metrics.results()

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        metrics = EpochMetrics()
        start_time = time.time()
        for images, targets in loader:
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
            self._save(epoch)
        test_metrics = self._evaluate(self.test_loader)
        if self.logger:
            self.logger.info(
                f"Test: loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} "
                f"samples={int(test_metrics['num_samples'])} time={test_metrics['time_sec']:.2f}s"
            )
        return test_metrics


