#!/bin/bash
# Extract metadata from OC20 IS2RE LMDB files
#
# This script extracts metadata from raw OC20 IS2RE data to create CSV files
# containing: system id, bulk id, miller index, shift, top, atomic numbers,
# lattice, coordinates, tags, reference energy, n_slab_layers, n_vacuum_layers
#
# Prerequisites:
#   - pip install fairchem-core pymatgen ase lmdb pandas
#   - Download OC20 IS2RE data: https://fair-chem.github.io/core/datasets/oc20.html
#
# Usage:
#   bash scripts/data/extract_metadata.sh <lmdb_path> <output_csv> [options]
#
# Arguments:
#   lmdb_path   - Path to OC20 IS2RE LMDB file or directory
#   output_csv  - Output CSV path (e.g., metadata/train_metadata.csv)
#
# Options (passed to Python script):
#   --start N        - Start index (default: 0)
#   --end N          - End index (default: all)
#   --num-workers N  - Number of parallel workers (default: CPU count)
#   --save-every N   - Save progress every N rows (default: 10000)
#
# Examples:
#   # Extract train split metadata
#   bash scripts/data/extract_metadata.sh \
#       dataset/oc20_raw/is2re/all/train/data.lmdb \
#       metadata/train_metadata.csv
#
#   # Extract with custom range
#   bash scripts/data/extract_metadata.sh \
#       dataset/oc20_raw/is2re/all/val_id/data.lmdb \
#       metadata/val_metadata.csv \
#       --end 10000

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"

LMDB_PATH="${1:?Error: LMDB path required}"
OUTPUT_CSV="${2:?Error: Output CSV path required}"
shift 2 || true

# Default mapping path
MAPPING_PATH="$PROJECT_ROOT/resources/oc20_data_mapping.pkl"

# Download mapping file if not present
if [ ! -f "$MAPPING_PATH" ]; then
    echo "Mapping file not found. Downloading..."
    bash "$PROJECT_ROOT/scripts/data/download_mapping.sh"
fi

echo "============================================"
echo "OC20 Metadata Extraction"
echo "============================================"
echo "LMDB path: $LMDB_PATH"
echo "Output CSV: $OUTPUT_CSV"
echo "Mapping file: $MAPPING_PATH"
echo "Additional args: $@"
echo ""

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_CSV")"

uv run python src/scripts/extract_metadata_oc20.py \
    --lmdb-path "$LMDB_PATH" \
    --mapping-path "$MAPPING_PATH" \
    --output-csv "$OUTPUT_CSV" \
    "$@"
