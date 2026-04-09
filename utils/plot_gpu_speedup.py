# utils/plot_gpu_speedup.py
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main(args):
    cpu_times = []
    gpu_times = []

    for seed in args.seeds:
        cpu_path = f"results/speedup_cpu_seed{seed}.json"
        gpu_path = f"results/speedup_gpu_seed{seed}.json"

        if not os.path.exists(cpu_path):
            print(f"Warning: {cpu_path} not found, skipping seed {seed}.")
            continue
        if not os.path.exists(gpu_path):
            print(f"Warning: {gpu_path} not found, skipping seed {seed}.")
            continue

        cpu_times.append(load_json(cpu_path)["total_time_seconds"])
        gpu_times.append(load_json(gpu_path)["total_time_seconds"])

    if not cpu_times or not gpu_times:
        raise FileNotFoundError("No speedup result files found.")

    cpu_mean = np.mean(cpu_times)
    cpu_std  = np.std(cpu_times)
    gpu_mean = np.mean(gpu_times)
    gpu_std  = np.std(gpu_times)

    speedup = cpu_mean / gpu_mean

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(
        ["CPU", "GPU"],
        [cpu_mean, gpu_mean],
        yerr=[cpu_std, gpu_std],
        capsize=8,
        color=["tab:blue", "tab:orange"],
        width=0.5,
        error_kw={"linewidth": 2},
    )

    # Annotate bar tops with mean ± std
    for bar, mean, std in zip(bars, [cpu_mean, gpu_mean], [cpu_std, gpu_std]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.5,
            f"{mean:.1f}s ± {std:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            )

    ax.set_ylabel("Total Wall-Clock Time (seconds)", fontsize=12)
    ax.set_title(
        f"GPU vs CPU Speedup — Sheaf-FMTL on Rotated MNIST\n"
        f"({len(cpu_times)} seeds, 10 rounds, 4 clients)  |  Speedup: {speedup:.2f}×",
        fontsize=12,
    )
    ax.set_ylim(0, max(cpu_mean + cpu_std, gpu_mean + gpu_std) * 1.3)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    save_path = "results/gpu_speedup_bar.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    print(f"CPU mean: {cpu_mean:.2f}s ± {cpu_std:.2f}s")
    print(f"GPU mean: {gpu_mean:.2f}s ± {gpu_std:.2f}s")
    print(f"Speedup:  {speedup:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1234])
    args = parser.parse_args()
    main(args)
