#!/bin/bash
#SBATCH --job-name=sheaf_fmtl_rmnist
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-rmnist-%j.out
#SBATCH --error=logs/slurm-rmnist-%j.err

source $PWD/venv/bin/activate

mkdir -p logs
mkdir -p results

NUM_ROUNDS=80
NUM_CLIENTS=10
NUM_ROTATIONS=10
ALPHA=0.0005
LAMBDA_REG=20.0
BATCH_SIZE=32
LOCAL_EPOCHS=1
GAMMA=1.0
ETA=0.00001
L2_STRENGTH=0.01
GRAPH_TYPE=complete
EDGE_PROBABILITY=1.0

SEEDS=(42 123 456 789 1234)

for SEED in "${SEEDS[@]}"; do
    echo "=== Seed $SEED: Run 1: local only (no communication) ==="
    python -u -m experiments.run_rotated_mnist \
        --num_clients $NUM_CLIENTS \
        --num_rotations $NUM_ROTATIONS \
        --lambda_reg 0 \
        --alpha $ALPHA \
        --eta $ETA \
        --gamma $GAMMA \
        --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS \
        --batch_size $BATCH_SIZE \
        --l2_strength $L2_STRENGTH \
        --graph_type $GRAPH_TYPE \
        --edge_probability $EDGE_PROBABILITY \
        --seed $SEED \
        --mode local \
        --save_results

    echo "=== Seed $SEED: Run 2: sheaf, eta=0 (no map update) ==="
    python -u -m experiments.run_rotated_mnist \
        --num_clients $NUM_CLIENTS \
        --num_rotations $NUM_ROTATIONS \
        --lambda_reg $LAMBDA_REG \
        --alpha $ALPHA \
        --eta 0.0 \
        --gamma $GAMMA \
        --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS \
        --batch_size $BATCH_SIZE \
        --l2_strength $L2_STRENGTH \
        --graph_type $GRAPH_TYPE \
        --edge_probability $EDGE_PROBABILITY \
        --seed $SEED \
        --mode sheaf \
        --save_results

    echo "=== Seed $SEED: Run 3: sheaf, eta=${ETA} (with map update) ==="
    python -u -m experiments.run_rotated_mnist \
        --num_clients $NUM_CLIENTS \
        --num_rotations $NUM_ROTATIONS \
        --lambda_reg $LAMBDA_REG \
        --alpha $ALPHA \
        --eta $ETA \
        --gamma $GAMMA \
        --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS \
        --batch_size $BATCH_SIZE \
        --l2_strength $L2_STRENGTH \
        --graph_type $GRAPH_TYPE \
        --edge_probability $EDGE_PROBABILITY \
        --seed $SEED \
        --mode sheaf \
        --save_results
done

echo "=== All runs complete. Generating plot... ==="
python -u -m utils.plot_compare_rmnist \
    --gamma $GAMMA \
    --lambda_reg $LAMBDA_REG \
    --eta $ETA
