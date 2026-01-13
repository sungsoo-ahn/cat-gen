#!/bin/bash
# Optimizer comparison: AdamW vs MuOn
# Runs both configurations in parallel on separate GPUs
#
# Model: Scale3 (hidden_dim=768, depth=24, heads=12)
# Dataset: Toy dataset (fast iteration)
# Epochs: 500
#
# Usage:
#   bash scripts/training/run_optimizer_comparison.sh
#
# Single optimizer (specify GPU):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/training/run_optimizer_comparison.sh adamw
#   CUDA_VISIBLE_DEVICES=1 bash scripts/training/run_optimizer_comparison.sh muon

set -e

# Get project root and set PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Check for single optimizer mode
if [ -n "$1" ]; then
    OPTIMIZER="$1"
    echo "Running single optimizer: $OPTIMIZER"
    echo ""

    if [ "$OPTIMIZER" == "adamw" ]; then
        echo "=== AdamW Optimizer ==="
        echo "Config: configs/optimizer/adamw.yaml"
        uv run python src/scripts/train.py configs/optimizer/adamw.yaml --overwrite
    elif [ "$OPTIMIZER" == "muon" ]; then
        echo "=== MuOn Optimizer ==="
        echo "Config: configs/optimizer/muon.yaml"
        uv run python src/scripts/train.py configs/optimizer/muon.yaml --overwrite
    else
        echo "Unknown optimizer: $OPTIMIZER"
        echo "Usage: $0 [adamw|muon]"
        exit 1
    fi

    exit 0
fi

# Parallel mode - run both optimizers on separate GPUs
echo "=== Optimizer Comparison: AdamW vs MuOn ==="
echo "Model: Scale3 (hidden_dim=768, depth=24, heads=12)"
echo "Dataset: Toy dataset"
echo "Epochs: 500"
echo ""

# Create output directories
mkdir -p data/catgen/optimizer/adamw
mkdir -p data/catgen/optimizer/muon

# Launch experiments in parallel
echo "[GPU 0] Starting AdamW baseline..."
CUDA_VISIBLE_DEVICES=0 uv run python src/scripts/train.py configs/optimizer/adamw.yaml --overwrite &
PID_ADAMW=$!

echo "[GPU 1] Starting MuOn optimizer..."
CUDA_VISIBLE_DEVICES=1 uv run python src/scripts/train.py configs/optimizer/muon.yaml --overwrite &
PID_MUON=$!

echo ""
echo "Experiments launched. PIDs:"
echo "  AdamW: $PID_ADAMW"
echo "  MuOn:  $PID_MUON"
echo ""
echo "Monitor with: nvidia-smi -l 1"
echo "Or check wandb dashboard for training curves"
echo ""

# Wait for both processes
echo "Waiting for experiments to complete..."
wait $PID_ADAMW && echo "[GPU 0] AdamW completed" || echo "[GPU 0] AdamW FAILED"
wait $PID_MUON && echo "[GPU 1] MuOn completed" || echo "[GPU 1] MuOn FAILED"

echo ""
echo "=== Comparison Complete ==="
echo "Check wandb for training curves comparison"
echo "Results saved to:"
echo "  - data/catgen/optimizer/adamw/"
echo "  - data/catgen/optimizer/muon/"
