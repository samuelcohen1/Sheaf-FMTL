import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import copy
import time
import json
from torch.utils.data import DataLoader

from datasets.rotated_mnist import RotatedMNIST
from models.cnn import SimpleCNN
from algorithms.sheaf_fmtl import SheafFMTL
from utils.graph_utils import generate_graph_by_type, get_graph_statistics
from utils.metrics import count_model_parameters, calculate_communication_bits

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Preparing Rotated MNIST dataset...")
    dataset = RotatedMNIST(num_clients=args.num_clients, num_rotations=args.num_rotations)
    client_train_datasets, client_test_datasets = dataset.prepare_data()

    print(f"Generating {args.graph_type} communication graph...")
    graph = generate_graph_by_type(
        args.num_clients,
        graph_type=args.graph_type,
        edge_probability=args.edge_probability,
        seed=args.seed
    )
    stats = get_graph_statistics(graph)
    print(f"Graph statistics: {stats}")

    print("Initializing client models...")
    client_models = [SimpleCNN(num_classes=10) for _ in range(args.num_clients)]

    train_loaders = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        for ds in client_train_datasets
    ]

    print("Initializing Sheaf-FMTL algorithm...")
    sheaf_fmtl = SheafFMTL(
        models=client_models,
        graph=graph,
        lambda_reg=args.lambda_reg,
        alpha=args.alpha,
        eta=args.eta,
        gamma=args.gamma
    )

    # Per-round timing lists (summed across all clients per round)
    local_times   = []  # shape: (num_rounds,)
    sheaf_times   = []
    restrict_times = []

    print(f"\nStarting timing experiment for {args.num_rounds} rounds...")

    for round_idx in range(args.num_rounds):
        round_local = 0.0
        round_sheaf = 0.0
        round_restrict = 0.0

        for client_id in range(args.num_clients):
            t0 = time.perf_counter()
            sheaf_fmtl.local_update(
                client_id,
                train_loaders[client_id],
                local_epochs=args.local_epochs,
                l2_strength=args.l2_strength
            )
            round_local += time.perf_counter() - t0

            t0 = time.perf_counter()
            sheaf_fmtl.sheaf_update(client_id)
            round_sheaf += time.perf_counter() - t0

            t0 = time.perf_counter()
            sheaf_fmtl.update_restriction_maps(client_id)
            round_restrict += time.perf_counter() - t0

        local_times.append(round_local)
        sheaf_times.append(round_sheaf)
        restrict_times.append(round_restrict)

        if round_idx % 10 == 0:
            print(f"Round {round_idx:3d}: "
                  f"local={round_local:.3f}s  "
                  f"sheaf={round_sheaf:.4f}s  "
                  f"restrict={round_restrict:.4f}s")

    results = {
        'args': vars(args),
        'graph_stats': stats,
        'local_times':    local_times,
        'sheaf_times':    sheaf_times,
        'restrict_times': restrict_times,
    }

    os.makedirs("results", exist_ok=True)
    save_path = "results/timing_breakdown.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTiming results saved to {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timing breakdown for Sheaf-FMTL on Rotated MNIST")

    parser.add_argument('--num_clients',    type=int,   default=8)
    parser.add_argument('--num_rotations',  type=int,   default=4)
    parser.add_argument('--lambda_reg',     type=float, default=0.001)
    parser.add_argument('--alpha',          type=float, default=0.0005)
    parser.add_argument('--eta',            type=float, default=0.00001)
    parser.add_argument('--gamma',          type=float, default=0.1)
    parser.add_argument('--num_rounds',     type=int,   default=100)
    parser.add_argument('--local_epochs',   type=int,   default=1)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--l2_strength',    type=float, default=0.01)
    parser.add_argument('--graph_type',     type=str,   default='complete',
                        choices=['erdos_renyi', 'small_world', 'scale_free', 'complete'])
    parser.add_argument('--edge_probability', type=float, default=0.15)
    parser.add_argument('--seed',           type=int,   default=42)

    args = parser.parse_args()
    main(args)
