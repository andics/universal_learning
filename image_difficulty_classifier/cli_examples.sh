# Example 1: Train with default settings (ResNet-50 fine-tuned)
# Output will automatically go to: output_RESNET50_TV_FINETUNE
python -m image_difficulty_classifier.train

# Example 2: Train ResNet-50 with frozen backbone (faster, less memory)
# Output will go to: output_RESNET50_TV_FROZEN
python -m image_difficulty_classifier.train \
  --freeze-backbone

# Example 3: Train with CLIP MLP instead of ResNet
# Output will go to: output_CLIP_MLP_VIT_B_32_openai  
python -m image_difficulty_classifier.train \
  --model-name clip_mlp

# Example 4: Train ResNet-101 for potentially better performance
# Output will go to: output_RESNET101_TV_FINETUNE
python -m image_difficulty_classifier.train \
  --model-name resnet101_tv

# Example 5: Custom learning rate for experimentation
python -m image_difficulty_classifier.train \
  --lr 5e-5 \
  --epochs 20


