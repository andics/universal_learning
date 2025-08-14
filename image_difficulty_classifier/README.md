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
python -m image_difficulty_classifier.train \
  --csv bars/imagenet_examples.csv \
  --output-dir image_difficulty_classifier/runs/clip_bin5 \
  --model-name clip_linear \
  --clip-backbone "ViT-B-32" \
  --clip-pretrained "openai" \
  --batch-size 128 \
  --epochs 10 \
  --lr 5e-4 \
  --weight-decay 0.01 \
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

- Uses CLIP (image tower) frozen by default, with a trainable linear head. You can set `--unfreeze-backbone` to fine-tune the backbone.
- Splits: 85% train, 5% val, 10% test (by index with a fixed seed).
- Checkpoints every 1/5th of an epoch by default; adjustable via `--checkpoint-every-fraction`.

### Optional path prefix replacement

If your `bars/imagenet_examples.csv` contains absolute paths beginning with `/om2/user/cheungb/datasets/imagenet_validation/val`, you can replace this prefix at runtime by passing `--path-prefix <your/new/prefix>`. When provided, the script will write a rewritten copy of the CSV to the package folder as `image_difficulty_classifier/imagenet_examples.csv` with the updated paths and will use it automatically.

Example:

```bash
python -m image_difficulty_classifier.train \
  --csv bars/imagenet_examples.csv \
  --path-prefix "D:/datasets/imagenet/val" \
  --output-dir image_difficulty_classifier/runs/clip_bin5 \
  --model-name clip_linear
```