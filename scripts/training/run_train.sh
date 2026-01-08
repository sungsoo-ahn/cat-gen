#!/bin/bash
# Train MinCatFlow model
#
# Usage:
#   bash scripts/training/run_train.sh
#   bash scripts/training/run_train.sh --overwrite
#   bash scripts/training/run_train.sh --debug

set -e

# Get project root (directory containing this script's parent)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"
uv run python src/scripts/train.py configs/default/default.yaml "$@"
