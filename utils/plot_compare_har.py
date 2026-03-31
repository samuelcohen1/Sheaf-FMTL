import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main(args):
    run_paths = {
        "Baseline (λ=0)":           f"results/sheaf_fmtl_har_gamma{args.gamma}_lambda0.0_eta{args.eta}.json",
        "Sheaf, η=0 (fixed maps)":  f"results/sheaf_fmtl_har_gamma{args.gamma}_lambda{args.lambda_reg}_eta0.0.json",
        f"Sheaf, η={args.eta}":     f"results/sheaf_fmtl_har_gamma{args.gamma}_lambda{args.lambda_reg}_eta{args.eta}.json",
    }

    results_dict = {}
    for label, path in run_paths.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        results_dict[label] = load_json(path)['history']

    if not results_dict:
        raise FileNotFoundError("No HAR result files found.")

    # One row per run: avg accuracy + per-client accuracy
    n_runs = len(results_dict)
    fig, axes = plt.subplots(n_runs, 2, figsize=(14, 5 * n_runs))
    if n_runs == 1:
        axes = [axes]  # ensure 2D indexing

    for row, (label, history) in enumerate(results_dict.items()):
        # history['test_accuracy'] is [round][client]
        acc_array = np.array(history['test_accuracy'])  # shape: (rounds, clients)
        avg_acc = acc_array.mean(axis=1)
        rounds = np.arange(len(avg_acc))
        bits_mb = [b / 1e6 for b in history['communication_bits']]

        n_clients = acc_array.shape[1]

        # --- Left: Accuracy vs Round ---
        ax = axes[row][0]
        for c in range(n_clients):
            ax.plot(rounds, acc_array[:, c], alpha=0.4, linewidth=1,
                    label=f"Client {c}")
        ax.plot(rounds, avg_acc, color='black', linewidth=2,
                linestyle='--', label="Average")
        ax.set_xlabel("Round")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"{label} — Accuracy vs. Round (γ={args.gamma})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True)

        # --- Right: Accuracy vs Communication ---
        ax = axes[row][1]
        for c in range(n_clients):
            ax.plot(bits_mb, acc_array[:, c], alpha=0.4, linewidth=1,
                    label=f"Client {c}")
        ax.plot(bits_mb, avg_acc, color='black', linewidth=2,
                linestyle='--', label="Average")
        ax.set_xlabel("Cumulative Communication (MB)")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"{label} — Accuracy vs. Communication (γ={args.gamma})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True)

    plt.tight_layout()
    save_path = f"results/har_comparison_gamma{args.gamma}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lambda_reg', type=float, default=20.0)
    parser.add_argument('--eta', type=float, default=0.01)
    args = parser.parse_args()
    main(args)
