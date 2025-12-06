
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Adjust path for imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fedavgm.dataset import partition, imagenette, cifar10

def plot_combined_distribution(x_train, y_train, x_test, y_test, num_classes, dataset_name, alphas, output_dir):
    """Generate a single figure containing subplots for each alpha."""
    num_plots = len(alphas)
    cols = 3
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", num_classes)
    
    for i, alpha in enumerate(alphas):
        ax = axes[i]
        print(f"Generating subplot for alpha={alpha}...")
        
        # Partition
        partitions = partition(x_train, y_train, 10, alpha) # Assuming 10 clients
        num_clients = len(partitions)
        
        # Count samples
        client_class_counts = np.zeros((num_clients, num_classes))
        for client_id, (x, y) in enumerate(partitions):
            unique, counts = np.unique(y, return_counts=True)
            for u, c in zip(unique, counts):
                client_class_counts[client_id, u] = c
        
        # Plot stacked bar on ax
        clients = np.arange(num_clients)
        bottom = np.zeros(num_clients)
        
        for class_id in range(num_classes):
            counts = client_class_counts[:, class_id]
            ax.bar(clients, counts, bottom=bottom, label=f"Class {class_id}" if i == 0 else "", color=colors[class_id])
            bottom += counts
            
        ax.set_title(f"Alpha = {alpha}")
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Number of Samples")
        if i == 0:
             # Only show legend for the first plot to avoid clutter, or put it outside
             pass

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.90)
    
    filename = os.path.join(output_dir, f"distribution_{dataset_name}_combined.png")
    plt.savefig(filename, dpi=150)
    print(f"Saved combined plot to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenette", choices=["cifar10", "imagenette"])
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet concentration parameter")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="imagenette_outputs", help="Directory to save plots")
    parser.add_argument("--all_alphas", action="store_true", help="Generate plots for all standard alphas")
    parser.add_argument("--combined", action="store_true", help="Generate a single combined figure for all alphas")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    alphas = [0.05, 0.1, 0.5, 1, 10, 100] if (args.all_alphas or args.combined) else [args.alpha]
    
    print(f"Loading data for {args.dataset}...")
    # Load data once
    if args.dataset == "imagenette":
        # Data is in fedavgm/fedavgm/data
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "imagenette2-160")
        x_train, y_train, _, _, _, num_classes = imagenette(10, (128, 128, 3), data_dir)
    else:
        x_train, y_train, _, _, _, num_classes = cifar10(10, (32, 32, 3))

    if args.combined:
        plot_combined_distribution(x_train, y_train, _, _, num_classes, args.dataset, alphas, args.output_dir)

if __name__ == "__main__":
    main()
