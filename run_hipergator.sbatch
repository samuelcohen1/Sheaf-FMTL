#!/bin/bash
#SBATCH --job-name=sheaf_fmtl            # job name
#SBATCH --partition=gpu                  # partition/queue
#SBATCH --gres=gpu:1                     # request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out            # stdout/stderr file

# load environment (adjust to your site)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sheaf_fmtl  # must have CUDA-enabled PyTorch

# optional: limit visible GPU
# export CUDA_VISIBLE_DEVICES=0

# run your script
python -m experiments.run_heterogeneous_cifar10 --num_rounds 5 --num_clients 10