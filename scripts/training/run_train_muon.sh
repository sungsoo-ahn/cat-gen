#!/bin/bash
# Train MinCatFlow model with MuOn optimizer
#
# Usage:
#   bash scripts/training/run_train_muon.sh
#   bash scripts/training/run_train_muon.sh --overwrite

set -e

# Get project root (directory containing this script's parent)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"
uv run python src/scripts/train.py configs/default/default_muon.yaml "$@"
