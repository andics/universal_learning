#!/usr/bin/env python3
"""
Convenience script to run single example training with sensible defaults.
"""

import subprocess
import sys
import os

def main():
    # Default arguments for the single example training
    cmd = [
        sys.executable, "train_grad.py",
        "--model_name", "efficientvit_b0.r224_in1k",
        "--model_csv_name", "efficientvit_base_0_224_classification_imagenet_1k",
        "--max_examples", "50",  # Train on 50 random wrong examples
        "--max_steps_per_example", "500",  # Up to 500 steps per example
        "--lr", "0.01",  # Higher learning rate for faster convergence
        "--weight_decay", "0.001"
    ]
    
    print("Running single example training with the following command:")
    print(" ".join(cmd))
    print()
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1

if __name__ == "__main__":
    exit(main())
