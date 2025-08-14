# Example 1: Train with default settings (CLIP MLP + ViT-L-14 + LAION weights)
# Output will automatically go to: output_CLIP_MLP_VIT_L_14_laion2b_s32b_b82k
python -m image_difficulty_classifier.train

# Example 2: Train with custom output directory (will still append model info)
# Output will go to: /custom/path_CLIP_MLP_VIT_L_14_laion2b_s32b_b82k
python -m image_difficulty_classifier.train \
  --output-dir /custom/path \
  --epochs 20

# Example 3: Train with linear head instead of MLP
# Output will go to: output_CLIP_LINEAR_VIT_L_14_laion2b_s32b_b82k
python -m image_difficulty_classifier.train \
  --model-name clip_linear

# Example 4: Train with different CLIP backbone
# Output will go to: output_CLIP_MLP_VIT_B_32_openai
python -m image_difficulty_classifier.train \
  --clip-backbone "ViT-B-32" \
  --clip-pretrained "openai"


