import json
import os
import numpy as np
import argparse

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main(args):
    results = []

    for lam in args.lambda_regs:
        for eta in args.etas:
            path = f"results/sheaf_fmtl_har_gamma{args.gamma}_lambda{lam}_eta{eta}_seed{args.seed}.json"
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping.")
                continue
            data = load_json(path)
            solo = data['final_accuracy']
            ens  = data.get('final_ensemble_accuracy', None)
            results.append({'lambda_reg': lam, 'eta': eta, 'solo': solo, 'ensemble': ens})

    if not results:
        print("No results found.")
        return

    results.sort(key=lambda x: x['solo'], reverse=True)

    print(f"\n{'Rank':<5} {'lambda_reg':<12} {'eta':<8} {'Solo Acc':<12} {'Ensemble Acc':<12}")
    print("-" * 52)
    for rank, r in enumerate(results, 1):
        ens_str = f"{r['ensemble']:.4f}" if r['ensemble'] is not None else "N/A"
        print(f"{rank:<5} {r['lambda_reg']:<12} {r['eta']:<8} {r['solo']:.4f}       {ens_str}")

    best = results[0]
    ens_str = f"{best['ensemble']:.4f}" if best['ensemble'] is not None else "N/A"
    print(f"\nBest: lambda_reg={best['lambda_reg']}, eta={best['eta']}, "
          f"solo={best['solo']:.4f}, ensemble={ens_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lambda_regs', type=float, nargs='+', required=True)
    parser.add_argument('--etas', type=float, nargs='+', required=True)
    args = parser.parse_args()
    main(args)
