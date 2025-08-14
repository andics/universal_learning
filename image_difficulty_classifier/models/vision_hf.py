from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import CLIPVisionModel, CLIPImageProcessor
except ImportError as e:
    raise ImportError("transformers is required for HF CLIP. Install via pip install transformers") from e


class HFCLIPLinearHead(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        unfreeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.vision: CLIPVisionModel = CLIPVisionModel.from_pretrained(backbone_name)
        embed_dim = self.vision.config.hidden_size
        self.classifier = nn.Linear(embed_dim, num_classes)
        if not unfreeze_backbone:
            for p in self.vision.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.vision(images)
        pooled = outputs.pooler_output  # [B, D]
        logits = self.classifier(pooled)
        return logits


def build_clip_hf_linear(num_classes: int, backbone: str = "openai/clip-vit-base-patch32", unfreeze_backbone: bool = False) -> nn.Module:
    return HFCLIPLinearHead(backbone_name=backbone, num_classes=num_classes, unfreeze_backbone=unfreeze_backbone)


