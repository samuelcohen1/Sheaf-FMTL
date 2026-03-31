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
        "Baseline (λ=0)":          f"results/sheaf_fmtl_har_gamma{args.gamma}_lambda0.0_eta{args.eta}.json",
        "Sheaf, η=0 (fixed maps)": f"results/sheaf_fmtl_har_gamma{args.gamma}_lambda{args.lambda_reg}_eta0.0.json",
        f"Sheaf, η={args.eta}":    f"results/sheaf_fmtl_har_gamma{args.gamma}_lambda{args.lambda_reg}_eta{args.eta}.json",
    }

    results_dict = {}
    for label, path in run_paths.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        results_dict[label] = load_json(path)['history']

    if not results_dict:
        raise FileNotFoundError("No HAR result files found.")

    # Determine number of clients from first result
    first_history = next(iter(results_dict.values()))
    acc_array = np.array(first_history['test_accuracy'])
    n_clients = acc_array.shape[1]

    fig, axes = plt.subplots(1, n_clients, figsize=(7 * n_clients, 5))
    if n_clients == 1:
        axes = [axes]

    for c in range(n_clients):
        ax = axes[c]
        for label, history in results_dict.items():
            acc_array = np.array(history['test_accuracy'])  # (iterations, clients)
            iterations = np.arange(len(acc_array))
            ax.plot(iterations, acc_array[:, c], linewidth=1.5, label=label)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Client {c} — Accuracy vs. Iteration (γ={args.gamma})")
        ax.legend(fontsize=9)
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
