from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)


def _freeze_all_but(module: nn.Module, trainable: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
    for p in trainable.parameters():
        p.requires_grad = True


def build_resnet50_tv(num_classes: int, unfreeze_backbone: bool = False) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if not unfreeze_backbone:
        _freeze_all_but(model, model.fc)
    return model


def build_resnet101_tv(num_classes: int, unfreeze_backbone: bool = False) -> nn.Module:
    weights = ResNet101_Weights.IMAGENET1K_V2
    model = resnet101(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if not unfreeze_backbone:
        _freeze_all_but(model, model.fc)
    return model


def build_vit_b_16_tv(num_classes: int, unfreeze_backbone: bool = False) -> nn.Module:
    # Prefer the stronger SWAG weights if available; fallback to IMAGENET1K_V1
    try:
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    except Exception:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    if not unfreeze_backbone:
        _freeze_all_but(model, model.heads)
    return model


def torchvision_transforms_for_weights(mean, std, image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def get_default_tv_transforms(model_name: str, image_size: int = 224) -> T.Compose:
    # Map names to weights to fetch recommended mean/std
    if model_name == "resnet50_tv":
        w = ResNet50_Weights.IMAGENET1K_V2
        return torchvision_transforms_for_weights(w.meta["mean"], w.meta["std"], image_size)
    if model_name == "resnet101_tv":
        w = ResNet101_Weights.IMAGENET1K_V2
        return torchvision_transforms_for_weights(w.meta["mean"], w.meta["std"], image_size)
    if model_name == "vit_b_16_tv":
        try:
            w = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        except Exception:
            w = ViT_B_16_Weights.IMAGENET1K_V1
        return torchvision_transforms_for_weights(w.meta["mean"], w.meta["std"], image_size)
    # Fallback to ImageNet defaults
    return torchvision_transforms_for_weights([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], image_size)


