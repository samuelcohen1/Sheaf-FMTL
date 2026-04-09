import json
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    path = "results/timing_breakdown.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No timing data found at {path}")

    with open(path) as f:
        data = json.load(f)

    local_times    = np.array(data['local_times'])
    sheaf_times    = np.array(data['sheaf_times'])
    restrict_times = np.array(data['restrict_times'])
    rounds = np.arange(len(local_times))

    labels  = ['Local Training', 'Sheaf Update', 'Restriction Map Update']
    arrays  = [local_times, sheaf_times, restrict_times]
    colors  = ['tab:blue', 'tab:orange', 'tab:green']

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, arr, color in zip(labels, arrays, colors):
        mean = arr.mean()
        std  = arr.std()
        ax.plot(rounds, arr, alpha=0.3, color=color, linewidth=0.8)
        ax.axhline(mean, color=color, linewidth=2.0, label=f"{label} (μ={mean:.4f}s, σ={std:.4f}s)")
        ax.axhspan(mean - std, mean + std, alpha=0.15, color=color)

    args = data.get('args', {})
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Wall-clock Time (s)", fontsize=12)
    ax.set_title(
        f"Sheaf-FMTL Timing Breakdown — Rotated MNIST\n"
        f"({args.get('num_clients', 8)} clients, complete graph, "
        f"γ={args.get('gamma', 0.1)}, {len(rounds)} rounds)",
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    save_path = "results/timing_breakdown.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
