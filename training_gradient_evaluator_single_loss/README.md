# Training Gradient Evaluator Single Loss

This is a modified version of the training_gradient_evaluator that trains on examples one at a time, recording the **sum of losses** until each example becomes correct.

## What it does

1. **Loads examples in difficulty order**: Uses the `imagenet_examples_ammended.csv` file which contains image paths sorted by universal difficulty (easiest first).

2. **Randomly samples wrong examples**: Instead of taking the first N examples, it:
   - Identifies all examples the model originally got wrong
   - Randomly samples from these wrong examples (for better statistical coverage)
   - Sorts the selected examples by difficulty for training order

3. **Trains one example at a time**: For each selected example:
   - Resets the model to original pretrained weights
   - Trains only on that single example until it gets it correct
   - **Sums all losses** during training until the example becomes correct
   - Records the total accumulated loss
   - Moves to the next example

4. **Stores results**: Creates a CSV file with:
   - Example index
   - Image path
   - Total loss accumulated until correct (-1 if never got correct)
   - Universal difficulty ranking (absolute position in difficulty order)

5. **Creates visualization**: Plots with:
   - X-axis: Total loss to get correct
   - Y-axis: Universal difficulty ranking (1=easiest)
   - Shows relationship between cumulative training loss and universal difficulty

## Usage

### Quick Start
```bash
cd training_gradient_evaluator_single_loss
python run_single_training.py
```

This will run with sensible defaults:
- resnet34 model
- 50 randomly sampled wrong examples
- Up to 500 steps per example
- Higher learning rate for faster convergence

### Custom Usage
```bash
python train_grad.py --model_name resnet18.a1_in1k --max_examples 100 --max_steps_per_example 1000
```

### Key Arguments
- `--model_name`: TIMM model name (default: resnet34.a3_in1k)
- `--max_examples`: Number of examples to train on (default: 1000)
- `--max_steps_per_example`: Max steps to train each example (default: 1000)
- `--lr`: Learning rate (default: 5e-6)
- `--model_csv_name`: Model name to look up in imagenet_models.csv

## Output Files

The script creates several output files in `outputs/{model_name}/`:

1. **single_example_results.csv**: Main results file with total loss for each example
2. **loss_vs_difficulty.png**: Scatter plot showing relationship between difficulty and cumulative loss  
3. **training_summary.json**: Summary statistics and all results
4. **train_single_{timestamp}.log**: Detailed training log

## Key Differences from Step-Based Version

- **Loss summation**: Accumulates all loss values during training instead of counting steps
- **Loss reset**: Total loss resets to 0 for each new example
- **Loss-based metrics**: All outputs focus on cumulative loss rather than step counts
- **Same structure**: Identical codebase structure with only the core metric changed

## Key Differences from Original

- **Single example training**: Trains one image at a time instead of batches
- **Random sampling**: Randomly samples from wrong examples instead of taking first N
- **Weight reset**: Resets to original pretrained weights before each example
- **Loss accumulation**: Sums losses until correct instead of counting steps
- **Difficulty ordering**: Uses universal difficulty ranking from imagenet_examples_ammended.csv
- **Updated visualization**: X-axis = total loss, Y-axis = universal difficulty ranking

## Requirements

Same as the original training_gradient_evaluator:
- PyTorch
- TIMM
- PIL/Pillow
- NumPy
- Matplotlib
- Access to ImageNet validation images