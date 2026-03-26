#!/bin/bash
#SBATCH --job-name=sheaf_fmtl_har
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-har-%j.out
#SBATCH --error=logs/slurm-har-%j.err

source $PWD/venv/bin/activate

mkdir -p logs
mkdir -p results

NUM_ROUNDS=200
NUM_CLIENTS=30
ALPHA=0.005
LAMBDA_REG=0.05
BATCH_SIZE=32
LOCAL_EPOCHS=1
GAMMA=0.3
ETA=0.01
L2_STRENGTH=0.01
GRAPH_TYPE=erdos_renyi
EDGE_PROBABILITY=0.2
TASK_INDEX_PATH=data/task_index.npy
DOWNSAMPLE_RATE=0.2
SEED=42

echo "=== Running Sheaf-FMTL on HAR dataset ==="
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

echo "=== HAR run complete ==="
