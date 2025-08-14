from typing import Optional

import torch
import torch.nn as nn


try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open_clip_torch is required for CLIP models. Install via pip install open_clip_torch"
    ) from e


class CLIPLinearHead(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: Optional[str],
        num_classes: int,
        unfreeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        # Returns (model, preprocess_train, preprocess_val); we keep only the model here.
        model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        self.clip_model = model

        # Freeze backbone unless explicitly requested otherwise
        if not unfreeze_backbone:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # Use a lazy linear so we do not need to know embed dim ahead of time.
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # open_clip expects normalized tensors; upstream dataset transforms handle normalization.
        with torch.autocast(device_type=images.device.type, enabled=False):
            # disable autocast inside encode_image if needed; outer context in trainer manages amp
            pass
        features = self.clip_model.encode_image(images)
        logits = self.classifier(features)
        return logits


class CLIPMLPHead(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: Optional[str],
        num_classes: int,
        unfreeze_backbone: bool = False,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Returns (model, preprocess_train, preprocess_val); we keep only the model here.
        model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        self.clip_model = model

        # Freeze backbone unless explicitly requested otherwise
        if not unfreeze_backbone:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # MLP classifier with dropout for better generalization
        self.classifier = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # open_clip expects normalized tensors; upstream dataset transforms handle normalization.
        with torch.autocast(device_type=images.device.type, enabled=False):
            # disable autocast inside encode_image if needed; outer context in trainer manages amp
            pass
        features = self.clip_model.encode_image(images)
        logits = self.classifier(features)
        return logits


def build_clip_linear_head(
    num_classes: int,
    *,
    clip_backbone: str = "ViT-B-32",
    clip_pretrained: Optional[str] = "openai",
    unfreeze_backbone: bool = False,
) -> nn.Module:
    return CLIPLinearHead(
        backbone=clip_backbone,
        pretrained=clip_pretrained,
        num_classes=num_classes,
        unfreeze_backbone=unfreeze_backbone,
    )


def build_clip_mlp_head(
    num_classes: int,
    *,
    clip_backbone: str = "ViT-B-32",
    clip_pretrained: Optional[str] = "openai",
    unfreeze_backbone: bool = False,
    hidden_dim: int = 512,
    dropout: float = 0.1,
) -> nn.Module:
    return CLIPMLPHead(
        backbone=clip_backbone,
        pretrained=clip_pretrained,
        num_classes=num_classes,
        unfreeze_backbone=unfreeze_backbone,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )



