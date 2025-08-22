#!/usr/bin/env python3
"""
Test script to verify the setup works correctly.
"""

import os
import sys
from pathlib import Path

# Ensure working directory and sys.path point to the Programming root
try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
    if path_main not in sys.path:
        sys.path.append(path_main)
    os.chdir(path_main)
except Exception:
    pass

def test_imports():
    """Test that all required imports work."""
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        
        import torch
        print("✓ PyTorch imported successfully")
        
        import timm
        print("✓ TIMM imported successfully")
        
        from training_gradient_evaluator_single.data import ImageNetWrongExamplesDataset, read_imagenet_paths, extract_synset_from_path
        print("✓ Local data modules imported successfully")
        
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_file_access():
    """Test that required files exist."""
    required_files = [
        "bars/imagenet_examples_ammended.csv",
        "bars/imagenet_models.csv",
        "bars/imagenet_synset_hierarchy.json",
        "bars/imagenet.npy"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ Found {file_path}")
        else:
            print(f"✗ Missing {file_path}")
            all_exist = False
    
    return all_exist

def test_model_loading():
    """Test that we can load a model."""
    try:
        import timm
        model = timm.create_model("efficientvit_b0.r224_in1k", pretrained=True)
        print("✓ Successfully loaded efficientvit_b0 model")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

def main():
    print("Testing training_gradient_evaluator_single setup...")
    print("=" * 50)
    
    # Test imports
    print("\n1. Testing imports:")
    imports_ok = test_imports()
    
    # Test file access
    print("\n2. Testing file access:")
    files_ok = test_file_access()
    
    # Test model loading
    print("\n3. Testing model loading:")
    model_ok = test_model_loading()
    
    # Summary
    print("\n" + "=" * 50)
    if imports_ok and files_ok and model_ok:
        print("✓ All tests passed! Setup is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
