# CatGen: Catalyst Structure Generation with Flow Matching

A PyTorch implementation of **CatGen** - a flow matching generative model for catalyst structure generation. This project generates realistic atomic structures of adsorbate+catalyst systems for materials science research.

## Overview

CatGen generates atomic structures including:
- **Primitive slabs**: Crystalline surface structures with periodic boundary conditions
- **Adsorbates**: Molecules (H, O, C, N compounds) bound to catalyst surfaces
- **Lattice parameters**: Cell dimensions (a, b, c) and angles (α, β, γ)
- **Supercell matrices**: Transformations to create full periodic systems

The model uses **flow matching** - a diffusion-based generative approach that learns to transform Gaussian noise into valid atomic structures by learning vector fields between source and target distributions.

## Features

- Flow matching with Diffusion Transformer (DiT) backbone
- Support for OC20 dataset processing
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

Download and process OC20 (Open Catalyst 2020) data:

```bash
# Download OC20 dataset (requires ~50GB disk space)
bash scripts/data/download_data.sh oc20

# Convert to CatGen format (outputs to processing_results/)
bash scripts/data/convert_oc20.sh train
bash scripts/data/convert_oc20.sh val_id
```

The processed data contains ~460K adsorbate+catalyst structures from DFT calculations.

### 2. Train a Model

```bash
# Train with full config
bash scripts/training/run_train.sh

# Quick test run (2 epochs, small model, uses processed OC20 data)
PYTHONPATH=. uv run python src/scripts/train.py configs/default/test.yaml
```

### 3. Generate Structures

```bash
# Generate structures using trained model
PYTHONPATH=. uv run python src/scripts/generate.py \
    configs/default/default.yaml \
    --checkpoint data/catgen/default/checkpoints/last.ckpt \
    --num_samples 10 \
    --sampling_steps 50
```

## Project Structure

```
cat-gen/
├── src/                          # Python source code
│   ├── catgen/                   # CatGen implementation
│   │   ├── data/                 # Data loading and processing
│   │   ├── models/               # Neural network architectures
│   │   │   ├── layers.py         # Encoder/Decoder layers
│   │   │   ├── transformers.py   # DiT blocks
│   │   │   └── loss/             # Loss functions and validation
│   │   └── module/               # PyTorch Lightning modules
│   │       ├── catgen_module.py  # Main training module
│   │       └── flow.py           # Flow matching logic
│   ├── scripts/                  # Entry point scripts
│   │   ├── train.py              # Training script
│   │   ├── generate.py           # Structure generation script
│   │   └── oc20_to_catgen.py     # OC20 to CatGen conversion
│   ├── helpers.py                # Utility functions
│   └── utils.py                  # Common utilities
├── configs/                      # YAML configuration files
│   └── default/
│       ├── default.yaml          # Full training config
│       └── test.yaml             # Quick test config
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

CatGen uses a three-stage architecture:

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

## Configuration

Key configuration options in `configs/*/default.yaml`:

```yaml
# Model architecture
model:
  atom_s: 256                    # Atom embedding dimension
  token_s: 256                   # Token embedding dimension
  flow_model_args:
    atom_encoder_depth: 2        # Encoder layers
    token_transformer_depth: 4   # Transformer layers
    atom_decoder_depth: 2        # Decoder layers

# Training
training:
  max_epochs: 20
  lr: 0.0001
  gradient_clip_val: 10.0

  # Loss weights
  prim_slab_coord_loss_weight: 1.0
  ads_coord_loss_weight: 1.0
  prim_virtual_loss_weight: 1.0
  supercell_virtual_loss_weight: 1.0
  scaling_factor_loss_weight: 1.0

# Validation
validation:
  sample_every_n_epochs: 2
  sampling_steps: 50
```

## Data Format

### Input (OC20)
- PyTorch Geometric Data objects in LMDB format
- Contains: atomic positions, atomic numbers, tags (0/1=surface, 2=adsorbate)

### Processed (CatGen LMDB)
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

This repository uses fixed random seeds for reproducible training. Set the same seed in your config to reproduce results exactly.

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
@article{catgen2024,
  title={CatGen: Catalyst Structure Generation with Flow Matching},
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
