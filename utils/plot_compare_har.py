import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main(args):
    seeds = [42, 123, 456, 789, 1234]

    run_configs = {
        "Baseline (λ=0)":          {"lambda": 0.0,             "eta": args.eta},
        "Sheaf, η=0 (fixed maps)": {"lambda": args.lambda_reg, "eta": 0.0},
        f"Sheaf, η={args.eta}":    {"lambda": args.lambda_reg, "eta": args.eta},
    }

    # For each config, load all seed files -> shape (n_seeds, n_iterations, n_clients)
    results_dict = {}
    for label, cfg in run_configs.items():
        seed_arrays = []
        for seed in seeds:
            path = f"results/sheaf_fmtl_har_gamma{args.gamma}_lambda{cfg['lambda']}_eta{cfg['eta']}_seed{seed}.json"
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping.")
                continue
            data = load_json(path)['history']['test_accuracy']
            seed_arrays.append(np.array(data))  # (n_iterations, n_clients)
        if seed_arrays:
            results_dict[label] = np.stack(seed_arrays, axis=0)  # (n_seeds, n_iterations, n_clients)

    if not results_dict:
        raise FileNotFoundError("No HAR result files found.")

    n_clients = next(iter(results_dict.values())).shape[2]

    fig, axes = plt.subplots(1, n_clients, figsize=(7 * n_clients, 5))
    if n_clients == 1:
        axes = [axes]

    for c in range(n_clients):
        ax = axes[c]
        for label, arr in results_dict.items():
            # arr shape: (n_seeds, n_iterations, n_clients)
            client_arr = arr[:, :, c]               # (n_seeds, n_iterations)
            mean = client_arr.mean(axis=0)
            std  = client_arr.std(axis=0)
            iterations = np.arange(mean.shape[0])

            line, = ax.plot(iterations, mean, linewidth=1.5, label=label)
            ax.fill_between(iterations, mean - std, mean + std,
                            alpha=0.2, color=line.get_color())

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
