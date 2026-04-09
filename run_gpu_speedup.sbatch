#!/bin/bash
#SBATCH --job-name=sheaf_gpu_speedup
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-gpu-speedup-%j.out
#SBATCH --error=logs/slurm-gpu-speedup-%j.err

source $PWD/venv/bin/activate

mkdir -p logs
mkdir -p results

NUM_ROUNDS=5
ALPHA=0.0051
NUM_CLIENTS=2
LAMBDA_REG=0.0011
BATCH_SIZE=512
LOCAL_EPOCHS=1
GRAPH_TYPE=complete
ETA=0.1
GAMMA=0.1

SEEDS=(42 123 456 789 1234)

echo "=== Running CPU runs ==="
for SEED in "${SEEDS[@]}"; do
    echo "--- CPU Seed $SEED ---"
    START=$(date +%s%N)
    CUDA_VISIBLE_DEVICES="" python -u -m experiments.run_rotated_mnist \
        --mode sheaf \
        --num_rounds $NUM_ROUNDS \
        --num_clients $NUM_CLIENTS \
        --num_rotations 2 \
        --gamma $GAMMA \
        --alpha $ALPHA \
        --eta $ETA \
        --lambda_reg $LAMBDA_REG \
        --batch_size $BATCH_SIZE \
        --local_epochs $LOCAL_EPOCHS \
        --graph_type $GRAPH_TYPE \
        --seed $SEED
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    echo "{\"device\": \"cpu\", \"seed\": $SEED, \"total_time_seconds\": $ELAPSED}" \
        > results/speedup_cpu_seed${SEED}.json
    echo "CPU Seed $SEED: ${ELAPSED}s"
done

echo "=== Running GPU runs ==="
for SEED in "${SEEDS[@]}"; do
    echo "--- GPU Seed $SEED ---"
    START=$(date +%s%N)
    python -u -m experiments.run_rotated_mnist \
        --mode sheaf \
        --num_rounds $NUM_ROUNDS \
        --num_clients $NUM_CLIENTS \
        --num_rotations 2 \
        --gamma $GAMMA \
        --alpha $ALPHA \
        --eta $ETA \
        --lambda_reg $LAMBDA_REG \
        --batch_size $BATCH_SIZE \
        --local_epochs $LOCAL_EPOCHS \
        --graph_type $GRAPH_TYPE \
        --seed $SEED
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    echo "{\"device\": \"gpu\", \"seed\": $SEED, \"total_time_seconds\": $ELAPSED}" \
        > results/speedup_gpu_seed${SEED}.json
    echo "GPU Seed $SEED: ${ELAPSED}s"
done

echo "=== All runs complete. Plotting speedup... ==="
python -u -m utils.plot_gpu_speedup --seeds 42 123 456 789 1234
