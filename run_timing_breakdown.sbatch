#!/bin/bash
#SBATCH --job-name=sheaf_timing
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-timing-%j.out
#SBATCH --error=logs/slurm-timing-%j.err

source $PWD/venv/bin/activate

mkdir -p logs
mkdir -p results

python -u -m experiments.run_timing_breakdown \
    --num_clients 8 \
    --num_rotations 8 \
    --gamma 0.1 \
    --alpha 0.0005 \
    --eta 0.00001 \
    --lambda_reg 0.001 \
    --batch_size 32 \
    --local_epochs 1 \
    --l2_strength 0.01 \
    --graph_type complete \
    --num_rounds 5 \
    --seed 42

echo "=== Run complete. Plotting... ==="
python -u -m utils.plot_timing_breakdown
