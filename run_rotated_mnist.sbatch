#!/bin/bash
#SBATCH --job-name=sheaf_fmtl_rmnist
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-rmnist-%j.out
#SBATCH --error=logs/slurm-rmnist-%j.err

source $PWD/venv/bin/activate

mkdir -p logs
mkdir -p results

GAMMA=0.1
NUM_ROUNDS=20
ALPHA=0.0051
NUM_CLIENTS=4
LAMBDA_REG=20.0
BATCH_SIZE=256
LOCAL_EPOCHS=1
GRAPH_TYPE=complete
ETA=0.1

SEEDS=(42 123 456 789 1234)

for SEED in "${SEEDS[@]}"; do
    echo "=== Seed $SEED: Running independent baseline ==="
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
        --graph_type $GRAPH_TYPE \
        --seed $SEED \
        --save_results

    echo "=== Seed $SEED: Running Sheaf-FMTL with eta=0.0 (fixed maps) ==="
    python -u -m experiments.run_rotated_mnist \
        --mode sheaf \
        --num_rounds $NUM_ROUNDS \
        --num_clients $NUM_CLIENTS \
        --num_rotations 4 \
        --gamma $GAMMA \
        --alpha $ALPHA \
        --eta 0.0 \
        --lambda_reg $LAMBDA_REG \
        --batch_size $BATCH_SIZE \
        --local_epochs $LOCAL_EPOCHS \
        --graph_type $GRAPH_TYPE \
        --seed $SEED \
        --save_results

    echo "=== Seed $SEED: Running Sheaf-FMTL with eta=$ETA (learned maps) ==="
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
        --graph_type $GRAPH_TYPE \
        --seed $SEED \
        --save_results
done

echo "=== All runs complete. Plotting... ==="
python -u -m utils.plot_compare_rmnist --gamma $GAMMA --lambda_reg $LAMBDA_REG --eta $ETA
