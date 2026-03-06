#!/bin/bash
#SBATCH --job-name=sheaf_fmtl
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source $PWD/venv/bin/activate

GAMMA=0.5
NUM_ROUNDS=50
NUM_CLIENTS=10
ETA_VALUES=(0.0001 0.001 0.01 0.1)

echo "=== Running independent baseline ==="
python -u -m experiments.run_rotated_mnist \
    --mode local \
    --num_rounds $NUM_ROUNDS \
    --num_clients $NUM_CLIENTS \
    --gamma $GAMMA \
    --save_results

for ETA in "${ETA_VALUES[@]}"; do
    echo "=== Running Sheaf-FMTL with eta=$ETA ==="
    python -u -m experiments.run_rotated_mnist \
        --mode sheaf \
        --num_rounds $NUM_ROUNDS \
        --num_clients $NUM_CLIENTS \
        --gamma $GAMMA \
        --eta $ETA \
        --lambda_reg 0.0002 \
        --save_results
done

echo "=== All runs complete. Plotting... ==="
python -u -m utils.plot_compare_mode --gamma $GAMMA
