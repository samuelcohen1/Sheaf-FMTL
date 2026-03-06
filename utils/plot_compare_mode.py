import json
import argparse
import os
import glob
import matplotlib.pyplot as plt
from utils.visualization import plot_communication_accuracy_tradeoff

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main(args):
    local_path = f"results/sheaf_fmtl_rmnist_local_gamma{args.gamma}.json"
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Baseline not found: {local_path}")

    local = load_json(local_path)

    # Find all sheaf runs for this gamma
    sheaf_paths = sorted(glob.glob(
        f"results/sheaf_fmtl_rmnist_sheaf_gamma{args.gamma}_eta*.json"
    ))
    if not sheaf_paths:
        raise FileNotFoundError(f"No sheaf results found for gamma={args.gamma}")

    results_dict = {'Independent (local)': local['history']}
    for path in sheaf_paths:
        data = load_json(path)
        eta = data['args']['eta']
        results_dict[f"Sheaf η={eta}"] = data['history']

    # --- Accuracy vs. Round ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, history in results_dict.items():
        rounds = range(len(history['test_accuracy']))
        axes[0].plot(rounds, history['test_accuracy'], label=label)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title(f"Accuracy vs. Round (γ={args.gamma})")
    axes[0].legend()
    axes[0].grid(True)

    # --- Accuracy vs. Communication ---
    for label, history in results_dict.items():
        bits_mb = [b / 1e6 for b in history['communication_bits']]
        axes[1].plot(bits_mb, history['test_accuracy'], label=label)
    axes[1].set_xlabel("Cumulative Communication (MB)")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title(f"Accuracy vs. Communication (γ={args.gamma})")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = f"results/eta_sweep_gamma{args.gamma}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
