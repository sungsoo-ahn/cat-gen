#!/bin/bash
# Download and prepare data for CatGen training
#
# CatGen Data Format
# ------------------
# CatGen uses a CUSTOM LMDB format that is DIFFERENT from standard OC20/OC22 LMDBs.
# Each sample contains:
#   - primitive_slab: ASE Atoms (decomposed primitive cell, NOT full supercell)
#   - supercell_matrix: numpy array (3x3) that transforms primitive -> supercell
#   - ads_atomic_numbers: numpy array of adsorbate atomic numbers
#   - ads_pos: numpy array (N, 3) of adsorbate positions
#   - ref_ads_pos: numpy array (N, 3) of reference adsorbate positions
#   - n_slab: int, number of slab layers
#   - n_vac: int, number of vacuum layers
#   - ref_energy: float, reference adsorption energy
#
# This preprocessing involves:
#   1. Finding the primitive cell from the supercell slab
#   2. Computing the supercell transformation matrix
#   3. Extracting adsorbate atoms from the catalyst system
#   4. Computing reference energies
#
# Data Options
# ------------
# Option 1: Use synthetic data for testing (already included)
#   - Run: uv run python src/scripts/create_synthetic_data.py
#   - Creates: dataset/train/dataset.lmdb, dataset/val_id/dataset.lmdb
#
# Option 2: Download OC20 IS2RE data (standard OC20 format, NOT CatGen format)
#   - This script downloads standard OC20 LMDBs
#   - These need conversion to CatGen format (see scripts/data/convert_oc20.sh)
#
# Option 3: Get preprocessed CatGen-format data from the authors
#   - Contact: https://github.com/sungsoo-ahn/cat-gen
#
# Usage:
#   bash scripts/data/download_data.sh [option]
#
# Options:
#   synthetic  - Create synthetic test data (default, fast)
#   oc20       - Download OC20 IS2RE data (large download)
#   info       - Show data format information only

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

OPTION="${1:-synthetic}"
DATA_DIR="${DATA_DIR:-dataset}"

echo "============================================"
echo "CatGen Data Preparation"
echo "============================================"
echo ""

case "$OPTION" in
    synthetic)
        echo "Creating synthetic test data..."
        echo "Output: $DATA_DIR/"
        echo ""
        uv run python src/scripts/create_synthetic_data.py \
            --output "$DATA_DIR" \
            --n-train 100 \
            --n-val 20
        echo ""
        echo "Done! Synthetic data created at: $DATA_DIR/"
        echo ""
        echo "Note: This is synthetic data for testing the pipeline."
        echo "For real training, you need CatGen-format data."
        ;;

    oc20)
        echo "Downloading OC20 IS2RE data..."
        echo ""
        echo "WARNING: This downloads standard OC20 LMDB format."
        echo "         It requires conversion to CatGen format."
        echo ""

        OC20_DIR="$DATA_DIR/oc20_raw"
        mkdir -p "$OC20_DIR"

        # Download IS2RE train data (smaller, LMDB format)
        echo "Downloading IS2RE train (10k split)..."
        wget -c https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz \
            -O "$OC20_DIR/is2res_train_val_test_lmdbs.tar.gz"

        echo ""
        echo "Extracting..."
        tar -xzf "$OC20_DIR/is2res_train_val_test_lmdbs.tar.gz" -C "$OC20_DIR"

        echo ""
        echo "Done! OC20 IS2RE data downloaded to: $OC20_DIR/"
        echo ""
        echo "NEXT STEPS:"
        echo "1. Convert to CatGen format using:"
        echo "   bash scripts/data/convert_oc20.sh train"
        echo "   bash scripts/data/convert_oc20.sh val_id"
        echo ""
        echo "2. Or use synthetic data for initial testing:"
        echo "   bash scripts/data/download_data.sh synthetic"
        ;;

    info)
        echo "CatGen LMDB Data Format"
        echo "======================="
        echo ""
        echo "Each sample in the LMDB contains:"
        echo ""
        echo "  primitive_slab: ASE Atoms object"
        echo "    - Primitive unit cell of the slab (NOT supercell)"
        echo "    - Contains atomic numbers and positions"
        echo "    - Cell vectors define primitive lattice"
        echo ""
        echo "  supercell_matrix: numpy array (3, 3)"
        echo "    - Transformation matrix: primitive -> supercell"
        echo "    - supercell = primitive @ supercell_matrix"
        echo ""
        echo "  ads_atomic_numbers: numpy array (N,)"
        echo "    - Atomic numbers of adsorbate atoms"
        echo ""
        echo "  ads_pos: numpy array (N, 3)"
        echo "    - Cartesian positions of adsorbate atoms"
        echo ""
        echo "  ref_ads_pos: numpy array (N, 3)"
        echo "    - Reference adsorbate positions (ground truth)"
        echo ""
        echo "  n_slab: int"
        echo "    - Number of slab atomic layers"
        echo ""
        echo "  n_vac: int"
        echo "    - Number of vacuum layers for z-scaling"
        echo ""
        echo "  ref_energy: float (optional)"
        echo "    - Reference adsorption energy in eV"
        echo ""
        echo "Key Differences from Standard OC20 LMDB:"
        echo "  - OC20 stores full catalyst+adsorbate structures"
        echo "  - CatGen decomposes into primitive slab + supercell matrix"
        echo "  - This decomposition enables flow matching on primitive cells"
        ;;

    *)
        echo "Unknown option: $OPTION"
        echo ""
        echo "Usage: bash scripts/data/download_data.sh [synthetic|oc20|info]"
        exit 1
        ;;
esac
