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
        "No Communication (local)":    {"mode": "local",  "eta": args.eta},
        "Sheaf, η=0 (fixed maps)":     {"mode": "sheaf",  "eta": 0.0},
        f"Sheaf, η={args.eta} (learned maps)": {"mode": "sheaf",  "eta": args.eta},
    }

    # results_dict[label]: shape (n_seeds, n_rounds) — mean accuracy per round
    results_dict = {}

    for label, cfg in run_configs.items():
        seed_arrays = []
        for seed in seeds:
            if cfg["mode"] == "local":
                path = f"results/sheaf_fmtl_rmnist_local_gamma{args.gamma}_seed{seed}.json"
            else:
                path = f"results/sheaf_fmtl_rmnist_sheaf_gamma{args.gamma}_eta{cfg['eta']}_seed{seed}.json"

            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping.")
                continue
            data = load_json(path)
            seed_arrays.append(np.array(data["history"]["test_accuracy"]))  # (n_rounds,)

        if seed_arrays:
            results_dict[label] = np.stack(seed_arrays, axis=0)  # (n_seeds, n_rounds)

    if not results_dict:
        raise FileNotFoundError("No rotated MNIST result files found.")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tab:gray", "tab:blue", "tab:orange"]

    for (label, arr), color in zip(results_dict.items(), colors):
        mean = arr.mean(axis=0)          # (n_rounds,)
        std  = arr.std(axis=0)           # (n_rounds,)
        iterations = np.arange(len(mean))

        line, = ax.plot(iterations, mean, linewidth=2.0, label=label, color=color)
        ax.fill_between(iterations, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Average Test Accuracy", fontsize=12)
    ax.set_title(
        f"Rotated MNIST — Sheaf-FMTL Comparison\n"
        f"(γ={args.gamma}, λ={args.lambda_reg}, {len(seeds)} seeds)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    save_path = f"results/rmnist_comparison_gamma{args.gamma}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=20.0)
    parser.add_argument("--eta", type=float, default=0.00001)
    args = parser.parse_args()
    main(args)
