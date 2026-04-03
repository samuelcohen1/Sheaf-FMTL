#!/bin/bash
#SBATCH --job-name=sheaf_har_grid
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-har-grid-%j.out
#SBATCH --error=logs/slurm-har-grid-%j.err

source $PWD/venv/bin/activate

mkdir -p logs
mkdir -p results

# Fixed constants (same as run_har.sbatch)
NUM_ROUNDS=80
NUM_CLIENTS=2
ALPHA=0.005
BATCH_SIZE=32
LOCAL_EPOCHS=1
GAMMA=1.0
L2_STRENGTH=0.01
GRAPH_TYPE=erdos_renyi
EDGE_PROBABILITY=1.0
TASK_INDEX_PATH=data/task_index.npy
DOWNSAMPLE_RATE=0.2
SEED=42

# Grid search values
LAMBDA_REGS=(0.0 1.0 5.0 10.0 20.0 50.0)
ETAS=(0.0 0.001 0.005 0.01 0.05 0.1)

for LAMBDA_REG in "${LAMBDA_REGS[@]}"; do
    for ETA in "${ETAS[@]}"; do
        echo "=== lambda_reg=${LAMBDA_REG}, eta=${ETA}, seed=${SEED} ==="
        python -u -m experiments.run_har \
            --num_clients $NUM_CLIENTS \
            --downsample_rate $DOWNSAMPLE_RATE \
            --task_index_path $TASK_INDEX_PATH \
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
            --save_results
    done
done

echo "=== Grid search complete. Ranking results... ==="
python -u -m utils.plot_grid_search_har \
    --gamma $GAMMA \
    --seed $SEED \
    --lambda_regs "${LAMBDA_REGS[@]}" \
    --etas "${ETAS[@]}"

echo "=== Done ==="
