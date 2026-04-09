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

    labels  = ['Local Training', 'Sheaf Update', 'Restriction Map Update']
    arrays  = [local_times, sheaf_times, restrict_times]
    colors  = ['tab:blue', 'tab:orange', 'tab:green']

    means = [arr.mean() for arr in arrays]
    stds  = [arr.std()  for arr in arrays]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, alpha=0.85,
                  error_kw={'elinewidth': 2, 'ecolor': 'black'})

    # Annotate each bar with mean ± std
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + max(means) * 0.01,
            f"{mean:.4f}s\n±{std:.4f}s",
            ha='center', va='bottom', fontsize=9
        )

    args = data.get('args', {})
    num_rounds = len(local_times)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Average Wall-clock Time per Round (s)", fontsize=12)
    ax.set_title(
        f"Sheaf-FMTL Timing Breakdown — Rotated MNIST\n"
        f"({args.get('num_clients', 8)} clients, complete graph, "
        f"γ={args.get('gamma', 0.1)}, {num_rounds} rounds)",
        fontsize=13
    )
    ax.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    save_path = "results/timing_breakdown.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
