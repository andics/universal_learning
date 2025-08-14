## Image Difficulty Classifier

Train a 5-class classifier that predicts the difficulty bin (0-4) of an ImageNet validation image, where bins are assigned by index in `bars/imagenet_examples.csv`:

- 0: indices [0, 9999]
- 1: indices [10000, 19999]
- 2: indices [20000, 29999]
- 3: indices [30000, 39999]
- 4: indices [40000, 49999]

### Install

```bash
cd Programming
pip install -r image_difficulty_classifier/requirements.txt
```

### Train

```bash
# Train with default settings (CLIP MLP + ViT-B-32 + OpenAI weights)
# Output automatically goes to: output_CLIP_MLP_VIT_B_32_openai
python -m image_difficulty_classifier.train

# Train with custom parameters
python -m image_difficulty_classifier.train \
  --csv bars/imagenet_examples.csv \
  --output-dir image_difficulty_classifier/runs/custom_run \
  --model-name clip_mlp \
  --clip-backbone "ViT-B-32" \
  --clip-pretrained "openai" \
  --batch-size 64 \
  --epochs 20 \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --num-workers 4 \
  --seed 42 \
  --checkpoint-every-fraction 0.2
```

Other model choices you can try:

- CLIP via Hugging Face image tower:
```bash
python -m image_difficulty_classifier.train \
  --csv bars/imagenet_examples.csv \
  --output-dir image_difficulty_classifier/runs/clip_hf_bin5 \
  --model-name clip_hf_linear \
  --hf-clip-backbone openai/clip-vit-base-patch32
```

- Torchvision ResNet-50:
```bash
python -m image_difficulty_classifier.train \
  --csv bars/imagenet_examples.csv \
  --output-dir image_difficulty_classifier/runs/resnet50_bin5 \
  --model-name resnet50_tv
```

- Torchvision ViT-B/16 (SWAG if available):
```bash
python -m image_difficulty_classifier.train \
  --csv bars/imagenet_examples.csv \
  --output-dir image_difficulty_classifier/runs/vit_b16_bin5 \
  --model-name vit_b_16_tv
```

Resuming is automatic if a latest checkpoint is present in `--output-dir`. You can also pass `--resume` explicitly.

### Notes

- **Default model**: CLIP MLP with ViT-B-32 backbone (frozen) + trainable MLP head with dropout
- **Output organization**: Automatically appends model info to output directory (e.g., `output_CLIP_MLP_VIT_B_32_openai`)
- **Learning rate**: Improved schedule with warmup (10% of training) and cosine decay
- **Splits**: 85% train, 5% val, 10% test (by index with a fixed seed)
- **Checkpoints**: Include epoch numbers and save every 1/5th of an epoch by default
- **Backbone fine-tuning**: Set `--unfreeze-backbone` to train the CLIP backbone (not recommended for small datasets)

 