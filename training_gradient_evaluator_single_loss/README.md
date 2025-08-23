# Training Gradient Evaluator Single Loss

This is a modified version of the training_gradient_evaluator that trains on examples one at a time, training until the **loss reaches within an epsilon of zero** and recording the number of SGD steps required.

## What it does

1. **Loads examples in difficulty order**: Uses the `imagenet_examples_ammended.csv` file which contains image paths sorted by universal difficulty (easiest first).

2. **Randomly samples wrong examples**: Instead of taking the first N examples, it:
   - Identifies all examples the model originally got wrong
   - Randomly samples from these wrong examples (for better statistical coverage)
   - Sorts the selected examples by difficulty for training order

3. **Trains one example at a time**: For each selected example:
   - Resets the model to original pretrained weights
   - Trains only on that single example until loss ≤ epsilon (default: 1e-3)
   - **Counts SGD steps** until loss reaches epsilon threshold
   - **Sums all losses** accumulated during training until epsilon
   - Records the total number of steps, loss sum, and final loss
   - **Logs each SGD step** with loss value and cumulative sum to individual CSV files
   - Moves to the next example

4. **Stores results**: Creates multiple output files:
   - **Main results CSV** with: example index, image path, total steps to epsilon, total loss sum, final loss, universal difficulty ranking
   - **Individual step logs** (in `step_logs/` directory): CSV files for each example containing step-by-step loss values and cumulative sums
   - **Summary JSON**: Overall statistics and results

5. **Creates dual visualizations**: 
   - **Plot 1**: SGD steps vs universal difficulty ranking
   - **Plot 2**: Loss sum vs universal difficulty ranking
   - Both show relationship between convergence metrics and universal difficulty

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
python train_grad.py --model_name resnet18.a1_in1k --max_examples 100 --max_steps_per_example 1000 --epsilon 1e-5
```

### Key Arguments
- `--model_name`: TIMM model name (default: efficientvit_b1.r224_in1k)
- `--epsilon`: Loss threshold to reach (default: 1e-3)
- `--max_examples`: Maximum number of examples to train on (default: 700)
- `--max_steps_per_example`: Maximum SGD steps per example (default: 10000)
- `--lr`: Learning rate for classical SGD (default: 0.01)
- `--model_csv_name`: Model name to look up in imagenet_models.csv

## Output Files

The script creates several output files in `outputs/{model_name}/`:

1. **`single_example_results.csv`**: Main results file with columns:
   - `example_index`: Sequential index of the example
   - `path`: Path to the image file
   - `total_steps_to_epsilon`: Number of SGD steps to reach epsilon (-1 if never reached)
   - `total_loss_sum`: Cumulative sum of all losses until epsilon reached
   - `final_loss`: Final loss value achieved
   - `universal_difficulty_rank`: Ranking in universal difficulty order (1=easiest)

2. **`step_logs/`**: Directory containing detailed step-by-step logs for each example:
   - `example_{idx}_{safe_path}_steps.csv`: Individual CSV files with columns:
     - `step`: SGD step number
     - `loss`: Loss value at that step
     - `cumulative_loss_sum`: Running total of all losses up to that step

3. **`steps_vs_difficulty.png`**: Scatter plot showing relationship between SGD steps and difficulty ranking

4. **`loss_sum_vs_difficulty.png`**: Scatter plot showing relationship between cumulative loss sum and difficulty ranking

5. **`training_summary.json`**: Summary statistics and complete results in JSON format

6. **`train_single_{timestamp}.log`**: Detailed training log with timestamps

## Key Changes from Original

- **Training criterion**: Changed from "until correct prediction" to "until loss ≤ epsilon"
- **Classical SGD**: Uses standard SGD with fixed learning rate (no adaptive algorithms like Adam/AdamW)
- **Single example training**: Handles BatchNorm layers properly for batch size = 1
- **Dual metrics**: Tracks both SGD steps AND cumulative loss sum until epsilon
- **Dual visualizations**: Creates two plots showing steps vs difficulty and loss sum vs difficulty
- **Enhanced logging**: Step-by-step logging includes both individual loss and cumulative sum
- **Epsilon parameter**: Configurable loss threshold (default: 1e-3)
- **Better convergence tracking**: Focus on optimization dynamics rather than classification accuracy

## Requirements

Same as the original training_gradient_evaluator:
- PyTorch
- TIMM
- PIL/Pillow
- NumPy
- Matplotlib
- Access to ImageNet validation images