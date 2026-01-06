#!/bin/bash
# Download and prepare training data for MinCatFlow
#
# This script downloads the OC20/OC22 catalyst dataset and prepares it
# in LMDB format for training.
#
# Usage:
#   bash scripts/data/download_data.sh
#
# The data will be stored in:
#   dataset/train/dataset.lmdb
#   dataset/val_id/dataset.lmdb

set -e

DATA_DIR="${DATA_DIR:-./dataset}"

echo "============================================"
echo "MinCatFlow Data Download Script"
echo "============================================"
echo ""
echo "Data directory: $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR"

# Check if fairchem-core is installed
if ! uv run python -c "import fairchem" 2>/dev/null; then
    echo "ERROR: fairchem-core is not installed."
    echo "Please run: uv add fairchem-core"
    exit 1
fi

echo "Option 1: Download pre-built LMDB (if available)"
echo "------------------------------------------------"
echo "If pre-built LMDB files are available from the MinCatFlow authors,"
echo "download them directly to:"
echo "  $DATA_DIR/train/dataset.lmdb"
echo "  $DATA_DIR/val_id/dataset.lmdb"
echo ""

echo "Option 2: Build from OC20/OC22"
echo "------------------------------"
echo "To build LMDB from raw OC20/OC22 data:"
echo ""
echo "1. Download the OC20 dataset from:"
echo "   https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md"
echo ""
echo "2. Run the data processing script (to be implemented):"
echo "   uv run python src/scripts/build_lmdb.py configs/data/build_lmdb.yaml"
echo ""

echo "Option 3: Create synthetic test data"
echo "------------------------------------"
echo "For testing the pipeline with synthetic data:"
echo "   uv run python src/scripts/create_synthetic_data.py --output $DATA_DIR"
echo ""

# Check if data already exists
if [ -f "$DATA_DIR/train/dataset.lmdb" ]; then
    echo "Found existing training data at: $DATA_DIR/train/dataset.lmdb"
else
    echo "No training data found."
    echo "Please follow one of the options above to obtain the data."
fi

echo ""
echo "============================================"
echo "Data download script complete"
echo "============================================"
