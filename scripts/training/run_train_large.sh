#!/bin/bash
# Train MinCatFlow model with large architecture (2x embeddings and depth)
#
# Usage:
#   bash scripts/training/run_train_large.sh
#   bash scripts/training/run_train_large.sh --overwrite

set -e

# Get project root (directory containing this script's parent)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"
uv run python src/scripts/train.py configs/default/default_large.yaml "$@"
