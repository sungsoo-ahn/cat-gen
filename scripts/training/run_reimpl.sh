#!/bin/bash
# Train using reimplementation
#
# Usage:
#   bash scripts/training/run_reimpl.sh
#   bash scripts/training/run_reimpl.sh --overwrite
#   bash scripts/training/run_reimpl.sh --debug

set -e

# Get project root (directory containing this script's parent)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"
uv run python src/scripts/train_reimpl.py configs/reimplementation/default.yaml "$@"
