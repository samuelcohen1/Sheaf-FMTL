#!/bin/bash
#SBATCH --job-name=sheaf_fmtl
#SBATCH --partition=hpg-b200        # replace if you prefer another valid partition from `sinfo`
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# activate venv in repo (use absolute path if created elsewhere)
source $PWD/venv/bin/activate

# run
# python -u -m experiments.run_rotated_mnist  --num_rounds 5 --num_clients 10

echo "=== Running independent baseline ==="
python -u -m experiments.run_rotated_mnist \
    --mode local \
    --num_rounds 5 \
    --num_clients 10 \
    --save_results \
    --gamma 0.02

echo "=== Running Sheaf-FMTL ==="
python -u -m experiments.run_rotated_mnist \
    --mode sheaf \
    --num_rounds 5 \
    --num_clients 10 \
    --save_results \
    --eta 0.01 \
    --gamma 0.02

echo "=== Both runs complete ==="

# ...existing code...
echo "=== Both runs complete ==="

python -u -m utils.plot_compare_mode --gamma 0.02   # not actually the right gamma