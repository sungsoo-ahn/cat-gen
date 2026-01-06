# CatGen: Catalyst Structure Generation with Flow Matching

A PyTorch implementation of **MinCatFlow** - a flow matching generative model for catalyst structure generation. This project generates realistic atomic structures of adsorbate+catalyst systems for materials science research.

## Overview

CatGen generates atomic structures including:
- **Primitive slabs**: Crystalline surface structures with periodic boundary conditions
- **Adsorbates**: Molecules (H, O, C, N compounds) bound to catalyst surfaces
- **Lattice parameters**: Cell dimensions (a, b, c) and angles (α, β, γ)
- **Supercell matrices**: Transformations to create full periodic systems

The model uses **flow matching** - a diffusion-based generative approach that learns to transform Gaussian noise into valid atomic structures by learning vector fields between source and target distributions.

## Features

- Flow matching with Diffusion Transformer (DiT) backbone
- Dynamic Nuclear Graph (DNG) mode for element prediction
- Support for OC20 dataset processing
- Parallel original and reimplementation for reproducibility comparison
- WandB integration for experiment tracking
- PyTorch Lightning training framework

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/cat-gen.git
cd cat-gen

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync
```

## Quick Start

### 1. Prepare Data

Download and process OC20 data:

```bash
# Download OC20 dataset (requires ~50GB disk space)
bash scripts/data/download_data.sh

# Extract metadata from raw LMDB files
bash scripts/data/extract_metadata.sh

# Convert to MinCatFlow format
bash scripts/data/process_oc20.sh train
bash scripts/data/process_oc20.sh val_id
```

### 2. Train a Model

```bash
# Train original implementation
bash scripts/training/run_original.sh

# Or train reimplementation
bash scripts/training/run_reimpl.sh

# For quick testing (2 epochs, small model)
PYTHONPATH=. uv run python src/scripts/train_original.py configs/original/test.yaml
```

### 3. Generate Structures

```bash
# Generate new catalyst structures
PYTHONPATH=. uv run python src/scripts/generate.py \
    configs/original/test.yaml \
    --checkpoint data/original/test_wandb/checkpoints/last.ckpt \
    --num_samples 10 \
    --sampling_steps 50
```

## Project Structure

```
cat-gen/
├── src/                          # Python source code
│   ├── original/                 # Original MinCatFlow implementation
│   │   ├── data/                 # Data loading and processing
│   │   ├── models/               # Neural network architectures
│   │   │   ├── layers.py         # Encoder/Decoder layers
│   │   │   ├── transformers.py   # DiT blocks
│   │   │   └── loss/             # Loss functions and validation
│   │   └── module/               # PyTorch Lightning modules
│   │       ├── effcat_module.py  # Main training module
│   │       └── flow.py           # Flow matching logic
│   ├── reimplementation/         # Reimplemented version (for reproducibility)
│   ├── scripts/                  # Entry point scripts
│   │   ├── train_original.py     # Training script (original)
│   │   ├── train_reimpl.py       # Training script (reimplementation)
│   │   ├── generate.py           # De novo generation script
│   │   └── process_oc20.py       # Data processing
│   ├── helpers.py                # Utility functions
│   └── utils.py                  # Common utilities
├── configs/                      # YAML configuration files
│   ├── original/
│   │   ├── default.yaml          # Full training config
│   │   └── test.yaml             # Quick test config
│   └── reimplementation/
│       ├── default.yaml
│       └── test.yaml
├── scripts/                      # Bash wrapper scripts
│   ├── data/                     # Data processing scripts
│   └── training/                 # Training scripts
├── dataset/                      # Input data
│   ├── oc20_raw/                 # Raw OC20 LMDB files
│   ├── train/                    # Processed training data
│   └── val_id/                   # Processed validation data
├── data/                         # Experiment outputs (gitignored)
├── docs/                         # Documentation
└── scratch/                      # Temporary work directory (gitignored)
```

## Model Architecture

MinCatFlow uses a three-stage architecture:

```
Input → AtomAttentionEncoder → TokenTransformer → AtomAttentionDecoder → Output
```

1. **AtomAttentionEncoder**: Jointly encodes primitive slab and adsorbate atoms
   - Input: noisy coordinates, lattice params, supercell matrix, timestep, elements
   - Output: 768-dim atom representations

2. **TokenTransformer**: Aggregates atom features to token level
   - 12-layer, 12-head transformer
   - Output: 768-dim token representations

3. **AtomAttentionDecoder**: Broadcasts tokens back to atom level
   - Outputs: coordinates, lattice parameters, supercell matrix, scaling factor
   - Optional: element predictions (DNG mode)

## Configuration

Key configuration options in `configs/*/default.yaml`:

```yaml
# Model architecture
model:
  atom_s: 768                    # Atom embedding dimension
  token_s: 768                   # Token embedding dimension
  flow_model_args:
    atom_encoder_depth: 6        # Encoder layers
    token_transformer_depth: 12  # Transformer layers
    atom_decoder_depth: 6        # Decoder layers
    dng: true                    # Enable element prediction

# Training
training:
  max_epochs: 1000
  lr: 0.0001
  gradient_clip_val: 10.0

  # Loss weights
  prim_slab_coord_loss_weight: 1.0
  ads_coord_loss_weight: 0.1
  length_loss_weight: 1.0
  angle_loss_weight: 0.01
  supercell_matrix_loss_weight: 1.0

# Validation
validation:
  sample_every_n_epochs: 5
  sampling_steps: 50
```

## Data Format

### Input (OC20)
- PyTorch Geometric Data objects in LMDB format
- Contains: atomic positions, atomic numbers, tags (0/1=surface, 2=adsorbate)

### Processed (MinCatFlow LMDB)
Each entry contains:
```python
{
    "sid": int,                      # System ID
    "primitive_slab": ase.Atoms,     # Primitive cell
    "supercell_matrix": np.array,    # (3, 3) transformation
    "n_slab": int,                   # Number of slab layers
    "n_vac": int,                    # Number of vacuum layers
    "ads_atomic_numbers": np.array,  # Adsorbate elements
    "ads_pos": np.array,             # Adsorbate positions (M, 3)
    "ref_energy": float              # Reference adsorption energy
}
```

## Training Outputs

Training creates the following structure in `data/<version>/<experiment>/`:

```
├── checkpoints/          # Model checkpoints
│   ├── last.ckpt
│   └── epoch=N-val_loss=X.ckpt
├── logs/                 # Training logs
│   └── wandb/            # WandB run data
├── figures/              # Visualizations
├── results/              # Metrics and results
└── config.yaml           # Saved configuration
```

## Validation Metrics

During training, the following metrics are computed:

| Metric | Description |
|--------|-------------|
| `val/prim_match_rate` | % of primitive cells matching ground truth |
| `val/prim_rmsd` | RMSD of primitive cell coordinates |
| `val/slab_match_rate` | % of full slabs matching |
| `val/slab_rmsd` | RMSD of full slab coordinates |
| `val/structural_validity` | % of valid structures (bond lengths) |
| `val/adsorption_energy` | Predicted adsorption energy (optional) |

## Reproducibility

This repository contains two parallel implementations:
- `src/original/` - Reference implementation
- `src/reimplementation/` - Verified reimplementation

Both produce **identical results** with the same random seed, confirming reproducibility.

## Dependencies

Key dependencies (see `pyproject.toml` for full list):
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- PyTorch Geometric 2.4+
- ASE (Atomic Simulation Environment)
- pymatgen
- LMDB
- WandB

## Citation

If you use this code, please cite:

```bibtex
@article{mincatflow2024,
  title={MinCatFlow: Flow Matching for Catalyst Structure Generation},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Add license information]

## Acknowledgments

- OC20 dataset from Open Catalyst Project
- Flow matching methodology
- DiT (Diffusion Transformer) architecture
