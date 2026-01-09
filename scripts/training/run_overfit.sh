#!/bin/bash
# Run overfitting test to verify model training is working correctly
#
# Expected behavior:
# - Training loss should steadily decrease
# - After ~200-500 epochs, loss should approach near-zero values
# - If loss plateaus at a high value, there may be a bug in the model
#
# Usage:
#   bash scripts/training/run_overfit.sh

set -e

echo "=== Overfitting Test ==="
echo "Training on 8 samples for 500 epochs"
echo "Expected: training loss should decrease to near-zero"
echo ""

uv run python src/scripts/train.py configs/default/overfit.yaml --overwrite

echo ""
echo "=== Test Complete ==="
echo "Check data/catgen/overfit_test/logs for training logs"
