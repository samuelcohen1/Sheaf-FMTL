import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from typing import List, Dict, Optional, Tuple

def plot_training_curves(history: Dict, title: str = "Training Progress", 
                        save_path: Optional[str] = None):
    """Plot training curves for accuracy/loss and communication"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy or loss
    if 'test_accuracy' in history:
        ax1.plot(history['test_accuracy'], 'b-', linewidth=2)
        ax1.set_ylabel('Test Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
    elif 'test_mse' in history:
        ax1.plot(history['test_mse'], 'r-', linewidth=2)
        ax1.set_ylabel('Test MSE', fontsize=12)
    
    ax1.set_xlabel('Communication Rounds', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{title} - Performance', fontsize=14)
    
    # Plot communication cost
    if 'communication_bits' in history:
        bits_mb = np.array(history['communication_bits']) / 1e6
        ax2.plot(bits_mb, 'g-', linewidth=2)
        ax2.set_ylabel('Cumulative Communication (MB)', fontsize=12)
        ax2.set_xlabel('Communication Rounds', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'{title} - Communication Cost', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_comparison(results_dict: Dict[str, Dict], metric: str = 'test_accuracy',
                   title: str = "Algorithm Comparison", save_path: Optional[str] = None):
    """Plot comparison of multiple algorithms"""
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for i, (algo_name, history) in enumerate(results_dict.items()):
        if metric in history:
            plt.plot(history[metric], 
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=2,
                    label=algo_name)
    
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_communication_accuracy_tradeoff(results_dict: Dict[str, Dict],
                                        save_path: Optional[str] = None):
    """Plot accuracy vs communication tradeoff"""
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (algo_name, history) in enumerate(results_dict.items()):
        if 'test_accuracy' in history and 'communication_bits' in history:
            bits_mb = np.array(history['communication_bits']) / 1e6
            plt.plot(bits_mb, history['test_accuracy'],
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    markersize=4,
                    linewidth=2,
                    label=algo_name)
    
    plt.xlabel('Transmitted Bits (MB)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Communication-Accuracy Tradeoff', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_client_distribution(client_datasets: List, dataset_name: str,
                           save_path: Optional[str] = None):
    """Plot data distribution across clients"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot number of samples per client
    num_samples = [len(dataset) for dataset in client_datasets]
    ax1.bar(range(len(num_samples)), num_samples, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Client ID', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title(f'{dataset_name} - Samples per Client', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot histogram of sample counts
    ax2.hist(num_samples, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax2.set_xlabel('Number of Samples', fontsize=12)
    ax2.set_ylabel('Number of Clients', fontsize=12)
    ax2.set_title(f'{dataset_name} - Distribution of Sample Counts', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_samples = np.mean(num_samples)
    std_samples = np.std(num_samples)
    ax2.axvline(mean_samples, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_samples:.1f}')
    ax2.axvline(mean_samples + std_samples, color='orange', linestyle=':', linewidth=2,
                label=f'±1 STD: {std_samples:.1f}')
    ax2.axvline(mean_samples - std_samples, color='orange', linestyle=':', linewidth=2)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_restriction_maps_analysis(P_dict: Dict[Tuple[int, int], torch.Tensor],
                                 save_path: Optional[str] = None):
    """Analyze and visualize restriction maps"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract Frobenius norms
    norms = []
    edges = []
    for (i, j), P_ij in P_dict.items():
        norm = torch.norm(P_ij, p='fro').item()
        norms.append(norm)
        edges.append((i, j))
    
    # Plot 1: Distribution of Frobenius norms
    ax = axes[0, 0]
    ax.hist(norms, bins=30, color='coral', edgecolor='darkred', alpha=0.7)
    ax.set_xlabel('Frobenius Norm', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Restriction Map Norms', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sorted norms
    ax = axes[0, 1]
    sorted_norms = sorted(norms, reverse=True)
    ax.plot(sorted_norms, 'b-', linewidth=2)
    ax.set_xlabel('Edge Index (sorted)', fontsize=12)
    ax.set_ylabel('Frobenius Norm', fontsize=12)
    ax.set_title('Sorted Restriction Map Norms', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Top connections
    ax = axes[1, 0]
    top_k = min(20, len(norms))
    sorted_indices = np.argsort(norms)[::-1][:top_k]
    top_edges = [edges[i] for i in sorted_indices]
    top_norms = [norms[i] for i in sorted_indices]
    
    y_pos = np.arange(len(top_edges))
    ax.barh(y_pos, top_norms, color='lightblue', edgecolor='darkblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{i}-{j}' for i, j in top_edges])
    ax.set_xlabel('Frobenius Norm', fontsize=12)
    ax.set_title(f'Top {top_k} Strongest Connections', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Compression ratio analysis
    ax = axes[1, 1]
    compression_ratios = []
    for (i, j), P_ij in P_dict.items():
        d_ij, d_i = P_ij.shape
        ratio = d_ij / d_i
        compression_ratios.append(ratio)
    
    ax.hist(compression_ratios, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax.set_xlabel('Compression Ratio (γ)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Compression Ratios', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def create_summary_table(results_list: List[Dict], metrics: List[str] = None):
    """Create a summary table of results"""
    if metrics is None:
        metrics = ['final_accuracy', 'total_communication_mb', 'total_time_seconds']
    
    # Extract data
    data = []
    for result in results_list:
        row = {
            'Algorithm': result.get('algorithm', 'Unknown'),
            'Dataset': result.get('dataset', 'Unknown')
        }
        
        for metric in metrics:
            if metric in result:
                value = result[metric]
                if isinstance(value, float):
                    row[metric] = f"{value:.4f}"
                else:
                    row[metric] = str(value)
            else:
                row[metric] = "N/A"
        
        data.append(row)
    
    # Print table
    headers = ['Algorithm', 'Dataset'] + metrics
    col_widths = [max(len(str(row.get(h, ''))) for row in data + [dict(zip(headers, headers))]) 
                  for h in headers]
    
    # Print header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in data:
        row_line = " | ".join(str(row.get(h, '')).ljust(w) for h, w in zip(headers, col_widths))
        print(row_line)

def save_results_to_json(results: Dict, filepath: str):
    """Save results to JSON file"""
    # Convert torch tensors and numpy arrays to lists
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")

def load_results_from_json(filepath: str) -> Dict:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results
