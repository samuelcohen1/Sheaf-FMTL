import json
import argparse
import os
from utils.visualization import plot_comparison, plot_communication_accuracy_tradeoff

def main(args):
    sheaf_path = f"results/sheaf_fmtl_rmnist_sheaf_gamma{args.gamma}.json"
    local_path = f"results/sheaf_fmtl_rmnist_local_gamma{args.gamma}.json"

    for path in [sheaf_path, local_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Results file not found: {path}")

    with open(sheaf_path) as f:
        sheaf = json.load(f)
    with open(local_path) as f:
        local = json.load(f)

    results_dict = {
        'Sheaf-FMTL': sheaf['history'],
        'Independent': local['history']
    }

    plot_comparison(
        results_dict,
        metric='test_accuracy',
        title=f'Sheaf-FMTL vs Independent (γ={args.gamma})',
        save_path=f'results/comparison_gamma{args.gamma}.png'
    )

    plot_communication_accuracy_tradeoff(
        results_dict,
        save_path=f'results/tradeoff_gamma{args.gamma}.png'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.01)
    args = parser.parse_args()
    main(args)