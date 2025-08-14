from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .vision_clip import build_clip_linear_head, build_clip_mlp_head
from .vision_torchvision import build_resnet50_tv, build_resnet101_tv, build_vit_b_16_tv
from .vision_hf import build_clip_hf_linear


_REGISTRY: Dict[str, Any] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


@register("clip_linear")
def _build_clip_linear(num_classes: int, **kwargs) -> nn.Module:
    return build_clip_linear_head(num_classes=num_classes, **kwargs)


@register("clip_mlp")
def _build_clip_mlp(num_classes: int, **kwargs) -> nn.Module:
    return build_clip_mlp_head(num_classes=num_classes, **kwargs)


@register("clip_hf_linear")
def _build_clip_hf_linear(num_classes: int, **kwargs) -> nn.Module:
    return build_clip_hf_linear(num_classes=num_classes, **kwargs)


@register("resnet50_tv")
def _build_resnet50(num_classes: int, **kwargs) -> nn.Module:
    return build_resnet50_tv(num_classes=num_classes, **kwargs)


@register("resnet101_tv")
def _build_resnet101(num_classes: int, **kwargs) -> nn.Module:
    return build_resnet101_tv(num_classes=num_classes, **kwargs)


@register("vit_b_16_tv")
def _build_vit_b16(num_classes: int, **kwargs) -> nn.Module:
    return build_vit_b_16_tv(num_classes=num_classes, **kwargs)


def list_models() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_model(name: str, *, num_classes: int, **kwargs) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    return _REGISTRY[name](num_classes=num_classes, **kwargs)
