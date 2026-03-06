#!/bin/bash
#SBATCH --job-name=sheaf_fmtl
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source $PWD/venv/bin/activate

GAMMA=0.01
NUM_ROUNDS=100
NUM_CLIENTS=4
ALPHA=0.005
LAMBDA_REG=0.001
BATCH_SIZE=256
LOCAL_EPOCHS=2
ETA_VALUES=(0.0001 0.001 0.01)

echo "=== Running independent baseline ==="
python -u -m experiments.run_rotated_mnist \
    --mode local \
    --num_rounds $NUM_ROUNDS \
    --num_clients $NUM_CLIENTS \
    --num_rotations 4 \
    --gamma $GAMMA \
    --alpha $ALPHA \
    --lambda_reg $LAMBDA_REG \
    --batch_size $BATCH_SIZE \
    --local_epochs $LOCAL_EPOCHS \
    --save_results

for ETA in "${ETA_VALUES[@]}"; do
    echo "=== Running Sheaf-FMTL with eta=$ETA ==="
    python -u -m experiments.run_rotated_mnist \
        --mode sheaf \
        --num_rounds $NUM_ROUNDS \
        --num_clients $NUM_CLIENTS \
        --num_rotations 4 \
        --gamma $GAMMA \
        --alpha $ALPHA \
        --eta $ETA \
        --lambda_reg $LAMBDA_REG \
        --batch_size $BATCH_SIZE \
        --local_epochs $LOCAL_EPOCHS \
        --save_results
done

echo "=== All runs complete. Plotting... ==="
python -u -m utils.plot_compare_mode --gamma $GAMMA
