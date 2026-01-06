#!/bin/bash
# Train using original MinCatFlow implementation
#
# Usage:
#   bash scripts/training/run_original.sh
#   bash scripts/training/run_original.sh --overwrite
#   bash scripts/training/run_original.sh --debug

set -e

# Get project root (directory containing this script's parent)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"
uv run python src/scripts/train_original.py configs/original/default.yaml "$@"
