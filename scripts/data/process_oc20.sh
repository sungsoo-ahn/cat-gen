#!/bin/bash
# Process OC20 metadata into MinCatFlow LMDB format
#
# This script processes the OC20 metadata CSV files to create
# MinCatFlow-format LMDB datasets with primitive slab decomposition.
#
# Usage:
#   bash scripts/data/process_oc20.sh [split] [options]
#
# Arguments:
#   split        - Dataset split: train, val, test (default: train)
#
# Options passed to Python script:
#   --start N    - Start index (default: 0)
#   --end N      - End index (default: all)
#   --num-workers N - Number of parallel workers (default: CPU count)
#   --chunksize N   - Chunk size for multiprocessing (default: 100)
#
# Examples:
#   bash scripts/data/process_oc20.sh train --end 1000  # First 1000 train samples
#   bash scripts/data/process_oc20.sh val               # All validation samples
#   bash scripts/data/process_oc20.sh train --num-workers 8

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"

SPLIT="${1:-train}"
shift || true  # Remove first arg if present

echo "============================================"
echo "OC20 to MinCatFlow Processing"
echo "============================================"
echo "Split: $SPLIT"
echo "Additional args: $@"
echo ""

uv run python src/scripts/process_oc20.py --split "$SPLIT" "$@"
