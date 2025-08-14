from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class EpochMetrics:
    loss_sum: float = 0.0
    correct: int = 0
    total: int = 0
    steps: int = 0

    def update(self, loss: float, logits: torch.Tensor, targets: torch.Tensor) -> None:
        batch_size = int(targets.numel())
        self.loss_sum += float(loss) * batch_size
        preds = torch.argmax(logits, dim=1)
        self.correct += int((preds == targets).sum().item())
        self.total += batch_size
        self.steps += 1

    def results(self) -> Dict[str, float]:
        avg_loss = self.loss_sum / max(self.total, 1)
        acc = self.correct / max(self.total, 1)
        return {"loss": avg_loss, "accuracy": acc, "num_samples": float(self.total), "num_steps": float(self.steps)}


