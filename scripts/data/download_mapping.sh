#!/bin/bash
# Download OC20 data mapping file from Open Catalyst Project
#
# This file contains metadata about each system in OC20:
# - bulk_mpid (Materials Project ID)
# - miller_index, shift, top (surface parameters)
# - ads_symbols, adsorption_site
#
# Source: https://fair-chem.github.io/catalysts/datasets/oc20.html

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${1:-resources}"
mkdir -p "$OUTPUT_DIR"

MAPPING_URL="https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl"
MAPPING_FILE="$OUTPUT_DIR/oc20_data_mapping.pkl"
EXPECTED_MD5="6b5d485019861f6e7efca38338375b61"

echo "============================================"
echo "OC20 Data Mapping Download"
echo "============================================"
echo "URL: $MAPPING_URL"
echo "Output: $MAPPING_FILE"
echo ""

if [ -f "$MAPPING_FILE" ]; then
    echo "File already exists. Verifying checksum..."
    if command -v md5sum &> /dev/null; then
        ACTUAL_MD5=$(md5sum "$MAPPING_FILE" | cut -d' ' -f1)
    elif command -v md5 &> /dev/null; then
        ACTUAL_MD5=$(md5 -q "$MAPPING_FILE")
    else
        echo "Warning: Cannot verify checksum (md5sum/md5 not found)"
        ACTUAL_MD5="$EXPECTED_MD5"
    fi

    if [ "$ACTUAL_MD5" = "$EXPECTED_MD5" ]; then
        echo "Checksum verified. File is up to date."
        exit 0
    else
        echo "Checksum mismatch. Re-downloading..."
        rm "$MAPPING_FILE"
    fi
fi

echo "Downloading mapping file..."
wget -O "$MAPPING_FILE" "$MAPPING_URL"

echo ""
echo "Verifying checksum..."
if command -v md5sum &> /dev/null; then
    ACTUAL_MD5=$(md5sum "$MAPPING_FILE" | cut -d' ' -f1)
elif command -v md5 &> /dev/null; then
    ACTUAL_MD5=$(md5 -q "$MAPPING_FILE")
else
    echo "Warning: Cannot verify checksum"
    ACTUAL_MD5="$EXPECTED_MD5"
fi

if [ "$ACTUAL_MD5" = "$EXPECTED_MD5" ]; then
    echo "Checksum verified successfully!"
else
    echo "Warning: Checksum mismatch!"
    echo "  Expected: $EXPECTED_MD5"
    echo "  Got: $ACTUAL_MD5"
fi

echo ""
echo "Download complete: $MAPPING_FILE"

# Show file info
python3 -c "
import pickle
with open('$MAPPING_FILE', 'rb') as f:
    data = pickle.load(f)
print(f'Mapping contains {len(data)} system entries')
sample_key = list(data.keys())[0]
print(f'Sample key: {sample_key}')
print(f'Sample value keys: {list(data[sample_key].keys())}')
" 2>/dev/null || echo "Could not inspect file (pickle load failed)"
