from typing import Optional

import torch
import timm


def create_default_model(model_name: Optional[str] = None, num_classes: int = 1000) -> torch.nn.Module:
	name = model_name or "mobilenetv3_small_050_224_lamb_imagenet_1k"
	model = timm.create_model(name, pretrained=True, num_classes=num_classes)
	return model

