#!/bin/bash
# Convert OC20 IS2RE LMDB directly to CatGen LMDB format
#
# Simplified pipeline that skips the intermediate CSV step.
#
# Prerequisites:
#   - OC20 IS2RE data downloaded
#   - oc20_data_mapping.pkl (auto-downloaded if missing)
#
# Usage:
#   bash scripts/data/convert_oc20.sh [split] [options]
#
# Arguments:
#   split - Dataset split: train, val_id, val_ood_ads, val_ood_cat, val_ood_both, test
#           (default: train)
#
# Options (passed to Python script):
#   --start N        - Start index (default: 0)
#   --end N          - End index (default: all)
#   --num-workers N  - Number of parallel workers (default: CPU count)
#
# Examples:
#   bash scripts/data/convert_oc20.sh train
#   bash scripts/data/convert_oc20.sh val_id --num-workers 8
#   bash scripts/data/convert_oc20.sh train --start 0 --end 10000

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

SPLIT="${1:-train}"
shift || true

# Paths
LMDB_PATH="dataset/oc20_raw/is2res_train_val_test_lmdbs/data/is2re/all/${SPLIT}/data.lmdb"
MAPPING_PATH="resources/oc20_data_mapping.pkl"
OUTPUT_PATH="processing_results/${SPLIT}/dataset.lmdb"

# Download mapping file if not present
if [ ! -f "$MAPPING_PATH" ]; then
    echo "Mapping file not found. Downloading..."
    bash "$PROJECT_ROOT/scripts/data/download_mapping.sh"
fi

# Check LMDB exists
if [ ! -f "$LMDB_PATH" ]; then
    echo "Error: LMDB file not found: $LMDB_PATH"
    echo ""
    echo "Please download OC20 IS2RE data first:"
    echo "  wget https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz"
    echo "  tar -xzf is2res_train_val_test_lmdbs.tar.gz -C dataset/oc20_raw/"
    exit 1
fi

echo "============================================"
echo "OC20 to CatGen Direct Conversion"
echo "============================================"
echo "Split: $SPLIT"
echo "Input: $LMDB_PATH"
echo "Output: $OUTPUT_PATH"
echo "Additional args: $@"
echo ""

uv run python src/scripts/oc20_to_catgen.py \
    --lmdb-path "$LMDB_PATH" \
    --mapping-path "$MAPPING_PATH" \
    --output-path "$OUTPUT_PATH" \
    "$@"
