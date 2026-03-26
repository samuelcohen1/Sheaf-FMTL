import json
import argparse
import os
import matplotlib.pyplot as plt

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main(args):
    run_paths = {
        "Baseline (λ=0)":           f"results/sheaf_fmtl_har_gamma{args.gamma}_lambda0_eta0.0.json",
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, history in results_dict.items():
        rounds = range(len(history['test_accuracy']))
        axes[0].plot(rounds, history['test_accuracy'], label=label)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title(f"HAR Accuracy vs. Round (γ={args.gamma})")
    axes[0].legend()
    axes[0].grid(True)

    for label, history in results_dict.items():
        bits_mb = [b / 1e6 for b in history['communication_bits']]
        axes[1].plot(bits_mb, history['test_accuracy'], label=label)
    axes[1].set_xlabel("Cumulative Communication (MB)")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title(f"HAR Accuracy vs. Communication (γ={args.gamma})")
    axes[1].legend()
    axes[1].grid(True)

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
