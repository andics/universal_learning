#!/usr/bin/env python3
"""
Script to analyze model accuracy metrics from trained_models folder and generate bar charts.
Creates two separate charts: one for models with accuracy < 50% and one for accuracy >= 50%.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

def extract_accuracy_metrics(trained_models_path):
    """
    Extract accuracy metrics from all model folders in trained_models directory.
    
    Args:
        trained_models_path (str): Path to the trained_models directory
        
    Returns:
        dict: Dictionary with model names as keys and accuracy values as values
    """
    accuracy_data = {}
    
    # Get the absolute path to trained_models
    if not os.path.isabs(trained_models_path):
        trained_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), trained_models_path)
    
    print(f"Scanning directory: {trained_models_path}")
    
    # Iterate through all subdirectories that start with "output_"
    for folder_name in os.listdir(trained_models_path):
        if folder_name.startswith("output_"):
            folder_path = os.path.join(trained_models_path, folder_name)
            
            if os.path.isdir(folder_path):
                test_metrics_path = os.path.join(folder_path, "test_metrics.json")
                
                if os.path.exists(test_metrics_path):
                    try:
                        with open(test_metrics_path, 'r') as f:
                            metrics = json.load(f)
                            
                        if 'accuracy' in metrics:
                            accuracy = metrics['accuracy']
                            accuracy_data[folder_name] = accuracy
                            print(f"Found {folder_name}: accuracy = {accuracy:.4f}")
                        else:
                            print(f"Warning: No 'accuracy' field found in {test_metrics_path}")
                            
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        print(f"Error reading {test_metrics_path}: {e}")
                else:
                    print(f"Warning: test_metrics.json not found in {folder_path}")
    
    return accuracy_data

def create_bar_charts(accuracy_data, output_dir):
    """
    Create two bar charts: one for models with accuracy < 50% and one for accuracy >= 50%.
    
    Args:
        accuracy_data (dict): Dictionary with model names and accuracy values
        output_dir (str): Directory to save the charts
    """
    # Separate models based on 50% threshold
    low_accuracy = {name: acc for name, acc in accuracy_data.items() if acc < 0.295}
    high_accuracy = {name: acc for name, acc in accuracy_data.items() if acc >= 0.50 and acc < 0.63}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up matplotlib for better-looking plots
    plt.style.use('default')
    
    def create_chart(data, title, filename, color):
        """Helper function to create a single bar chart."""
        if not data:
            print(f"No data for {title}")
            return
            
        # Sort by accuracy for better visualization
        sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(sorted_data) * 0.8), 8))
        
        # Extract names and accuracies
        names = list(sorted_data.keys())
        accuracies = [acc * 100 for acc in sorted_data.values()]  # Convert to percentage
        
        # Create bar chart
        bars = ax.bar(range(len(names)), accuracies, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('Model Names', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        
        # Add value labels on top of bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add grid for better readability
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Set y-axis limits
        max_acc = max(accuracies) if accuracies else 100
        ax.set_ylim(0, min(max_acc + 10, 100))
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved chart: {filepath}")
        
        # Close the figure to free memory
        plt.close()
    
    # Create charts
    create_chart(low_accuracy, 
                'Models with Accuracy < 50%', 
                'low_accuracy_models.png', 
                '#ff6b6b')
    
    create_chart(high_accuracy, 
                'Models with Accuracy ≥ 50%', 
                'high_accuracy_models.png', 
                '#4ecdc4')
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total models processed: {len(accuracy_data)}")
    print(f"Models with accuracy < 50%: {len(low_accuracy)}")
    print(f"Models with accuracy ≥ 50%: {len(high_accuracy)}")
    
    if low_accuracy:
        avg_low = np.mean(list(low_accuracy.values()))
        print(f"Average accuracy for low-performing models: {avg_low:.2%}")
    
    if high_accuracy:
        avg_high = np.mean(list(high_accuracy.values()))
        print(f"Average accuracy for high-performing models: {avg_high:.2%}")

def main():
    """Main function to run the analysis."""
    # Get the current absolute time for folder naming
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level to Programming folder
    trained_models_path = os.path.join(project_root, "trained_models")
    output_dir = os.path.join(script_dir, f"images_{current_time}")
    
    print("Model Accuracy Analyzer")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Trained models path: {trained_models_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Extract accuracy metrics
    print("Extracting accuracy metrics...")
    accuracy_data = extract_accuracy_metrics(trained_models_path)
    
    if not accuracy_data:
        print("No accuracy data found. Please check if the trained_models folder exists and contains output_ folders with test_metrics.json files.")
        return
    
    print(f"\nFound {len(accuracy_data)} models with accuracy data.")
    print()
    
    # Create bar charts
    print("Generating bar charts...")
    create_bar_charts(accuracy_data, output_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
