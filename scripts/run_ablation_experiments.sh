#!/bin/bash
# Run 4 ablation experiments in parallel on 4 GPUs
#
# Experiments:
# GPU 0: baseline     - all improvements OFF
# GPU 1: +timesteps   - exponential timesteps only
# GPU 2: +ema         - exponential timesteps + EMA
# GPU 3: +all         - all improvements ON

set -e

echo "Starting 4 ablation experiments on 4 GPUs..."
echo "=============================================="
echo ""
echo "GPU 0: baseline     (all OFF)"
echo "GPU 1: +timesteps   (exponential timesteps)"
echo "GPU 2: +ema         (timesteps + EMA)"
echo "GPU 3: +all         (all improvements)"
echo ""

# Run all experiments in parallel
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python src/scripts/train.py \
    configs/experiments/ablation_baseline.yaml --overwrite &
PID0=$!

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python src/scripts/train.py \
    configs/experiments/ablation_timesteps.yaml --overwrite &
PID1=$!

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. uv run python src/scripts/train.py \
    configs/experiments/ablation_ema.yaml --overwrite &
PID2=$!

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. uv run python src/scripts/train.py \
    configs/experiments/ablation_all.yaml --overwrite &
PID3=$!

echo "Launched all experiments. PIDs: $PID0, $PID1, $PID2, $PID3"
echo ""
echo "Monitor progress:"
echo "  - WandB: https://wandb.ai (project: CatGen, filter by ablation tag)"
echo "  - Logs: tail -f data/experiments/ablation_*/logs/*"
echo ""

# Wait for all to complete
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "=============================================="
echo "All experiments completed!"
echo ""
echo "Results saved to:"
echo "  - data/experiments/ablation_baseline/"
echo "  - data/experiments/ablation_timesteps/"
echo "  - data/experiments/ablation_ema/"
echo "  - data/experiments/ablation_all/"
