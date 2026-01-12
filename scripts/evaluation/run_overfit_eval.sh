#!/bin/bash
# Evaluate overfitting experiment
#
# Usage:
#   bash scripts/evaluation/run_overfit_eval.sh [checkpoint_path]

set -e

CHECKPOINT="${1:-data/catgen/overfit_test/checkpoints/last.ckpt}"

echo "=== Overfit Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo ""

uv run python src/scripts/evaluate_overfit.py \
    configs/default/overfit.yaml \
    --checkpoint "$CHECKPOINT" \
    --wandb-project CatGen \
    --wandb-run-name "overfit_eval_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=== Evaluation Complete ==="
