#!/bin/bash
# Train overfitting experiment with random prior initialization
# This is a harder test - model must learn flow from ANY random starting point

set -e

echo "=== Overfitting with Random Prior ==="
echo "This is a harder test than fixed prior overfitting"
echo ""

uv run python src/scripts/train.py configs/default/overfit_random_prior.yaml

echo ""
echo "=== Training Complete ==="
