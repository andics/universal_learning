python -m image_difficulty_classifier.train \
  --csv Programming/bars/imagenet_examples.csv \
  # Optionally replace the default ImageNet validation prefix with a local path
  # --path-prefix "D:/datasets/imagenet/val" \
  --output-dir Programming/image_difficulty_classifier/runs/clip_bin5 \
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


