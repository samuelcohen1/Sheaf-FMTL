#!/bin/bash
#SBATCH --job-name=sheaf_fmtl
#SBATCH --partition=hpg-default        # replace if you prefer another valid partition from `sinfo`
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# activate venv in repo (use absolute path if created elsewhere)
source $PWD/venv/bin/activate

# run
python -m experiments.run_heterogeneous_cifar10 --device cuda --num_rounds 5 --num_clients 10