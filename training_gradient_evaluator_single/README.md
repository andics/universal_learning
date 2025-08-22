# Training Gradient Evaluator Single

This is a modified version of the training_gradient_evaluator that trains on examples one at a time, in order of universal difficulty ranking.

## What it does

1. **Loads examples in difficulty order**: Uses the `imagenet_examples_ammended.csv` file which contains image paths sorted by universal difficulty (easiest first).

2. **Trains one example at a time**: For each example:
   - Resets the model to original pretrained weights
   - Trains only on that single example until it gets it correct
   - Records the number of steps it took to get it right
   - Moves to the next example

3. **Stores results**: Creates a CSV file with:
   - Example index
   - Image path
   - Steps to get correct (-1 if never got correct)
   - Universal difficulty ranking

4. **Creates visualization**: Plots steps vs universal difficulty ranking to show the relationship.

## Usage

### Quick Start
```bash
cd training_gradient_evaluator_single
python run_single_training.py
```

This will run with sensible defaults:
- efficientvit_b0 model
- First 20 examples
- Up to 500 steps per example
- Higher learning rate for faster convergence

### Custom Usage
```bash
python train_grad.py --model_name resnet18.a1_in1k --max_examples 50 --max_steps_per_example 1000
```

### Key Arguments
- `--model_name`: TIMM model name (default: efficientvit_b0.r224_in1k)
- `--max_examples`: Number of examples to train on (default: 10)
- `--max_steps_per_example`: Max steps to train each example (default: 1000)
- `--lr`: Learning rate (default: 0.004)
- `--model_csv_name`: Model name to look up in imagenet_models.csv

## Output Files

The script creates several output files in `outputs/{model_name}/`:

1. **single_example_results.csv**: Main results file with steps for each example
2. **steps_vs_difficulty.png**: Scatter plot showing relationship between difficulty and training steps  
3. **training_summary.json**: Summary statistics and all results
4. **train_single_{timestamp}.log**: Detailed training log

## Key Differences from Original

- **Single example training**: Trains one image at a time instead of batches
- **Weight reset**: Resets to original pretrained weights before each example
- **Difficulty ordering**: Uses universal difficulty ranking from imagenet_examples_ammended.csv
- **Step counting**: Records exact number of steps to get each example correct
- **Simplified output**: Focuses on steps vs difficulty relationship

## Requirements

Same as the original training_gradient_evaluator:
- PyTorch
- TIMM
- PIL/Pillow
- NumPy
- Matplotlib
- Access to ImageNet validation images
