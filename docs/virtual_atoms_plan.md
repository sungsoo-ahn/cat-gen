# Virtual Atoms Architecture Plan

## Overview
This document outlines the plan to modify the cat-gen codebase to represent lattice and supercell information using virtual atoms that live in the same space as real atoms.

## Current Architecture Summary

### Current Atom Types
1. **Primitive Slab Atoms** (N real atoms)
   - Atomic numbers: [N]
   - Positions: [N, 3] in Cartesian coordinates

2. **Adsorbate Atoms** (M real atoms)
   - Atomic numbers: [M]
   - Positions: [M, 3] in Cartesian coordinates

### Current Lattice/Supercell Representation
- **Lattice**: 6 parameters (a, b, c, alpha, beta, gamma)
- **Supercell Matrix**: 3x3 transformation matrix
- **Scaling Factor**: 1 scalar (z-direction vacuum scaling)

These are represented as **separate tensors** and embedded independently in the model.

---

## Proposed Architecture

### Design Decision: 6 Virtual Atoms

**Architecture**:
- 3 virtual atoms for **primitive lattice vectors** (a, b, c)
- 3 virtual atoms for **supercell lattice vectors** (a', b', c')

**Key Design Principle**: **Minimal data processing changes**
- LMDB storage format remains unchanged (still stores lattice params + supercell matrix)
- Virtual atom coordinates are computed **on-the-fly** during batch collation
- Conversion functions: lattice_params → primitive_virtual_coords, (lattice + supercell_matrix) → supercell_virtual_coords

### Proposed Atom Types
1. **Primitive Cell Atoms** (N real atoms)
   - Type: Real atoms with atomic numbers 1-100
   - Positions: [N, 3] in Cartesian coordinates
   - Features: Element one-hot + padding mask

2. **Adsorbate Atoms** (M real atoms)
   - Type: Real atoms with atomic numbers 1-100
   - Positions: [M, 3] in Cartesian coordinates
   - Features: Element one-hot + ref position + binding atom + padding mask

3. **Primitive Lattice Virtual Atoms** (3 virtual atoms)
   - Type: Virtual atoms with special ID (0, 1, 2)
   - Positions: [3, 3] where each row is a primitive lattice vector (a, b, c)
   - Features: Virtual atom ID one-hot encoding
   - Computed from: lattice parameters (a, b, c, α, β, γ)
   - **No padding**: Always present (no variable length)

4. **Supercell Lattice Virtual Atoms** (3 virtual atoms)
   - Type: Virtual atoms with special ID (3, 4, 5)
   - Positions: [3, 3] where each row is a supercell lattice vector (a', b', c')
   - Features: Virtual atom ID one-hot encoding
   - Computed from: primitive lattice vectors @ supercell_matrix.T
   - **No padding**: Always present (no variable length)

---

## Detailed Conversion Functions

### Core Conversion Logic

#### Forward Conversion (Data → Model)

```python
import numpy as np
import torch
from pymatgen.core import Lattice

def lattice_params_to_vectors(lattice_params: np.ndarray) -> np.ndarray:
    """
    Convert lattice parameters to lattice vectors (row vectors).

    Args:
        lattice_params: [6] array of (a, b, c, alpha, beta, gamma)
                       lengths in Angstrom, angles in degrees

    Returns:
        lattice_vectors: [3, 3] array where each row is a lattice vector [a, b, c]
    """
    a, b, c, alpha, beta, gamma = lattice_params
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    # lattice.matrix is [3, 3] with columns as lattice vectors
    # We want row vectors, so transpose
    return lattice.matrix.T

def compute_virtual_coords(
    lattice_params: torch.Tensor,
    supercell_matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute primitive and supercell virtual atom coordinates.

    Args:
        lattice_params: [B, 6] tensor of lattice parameters
        supercell_matrix: [B, 3, 3] tensor of supercell transformation matrices

    Returns:
        prim_virtual_coords: [B, 3, 3] primitive lattice vectors (row vectors)
        supercell_virtual_coords: [B, 3, 3] supercell lattice vectors (row vectors)
    """
    B = lattice_params.shape[0]
    prim_virtual_coords = torch.zeros(B, 3, 3)

    # Convert each sample's lattice params to vectors
    for i in range(B):
        params = lattice_params[i].cpu().numpy()
        vectors = lattice_params_to_vectors(params)
        prim_virtual_coords[i] = torch.from_numpy(vectors)

    # Move to same device as input
    prim_virtual_coords = prim_virtual_coords.to(lattice_params.device)

    # Compute supercell vectors: supercell = prim @ supercell_matrix^T
    supercell_virtual_coords = torch.bmm(prim_virtual_coords, supercell_matrix.transpose(-2, -1))

    return prim_virtual_coords, supercell_virtual_coords
```

#### Backward Conversion (Model → Assembly)

```python
def lattice_vectors_to_params(lattice_vectors: np.ndarray) -> np.ndarray:
    """
    Convert lattice vectors to lattice parameters.

    Args:
        lattice_vectors: [3, 3] array where each row is a lattice vector

    Returns:
        lattice_params: [6] array of (a, b, c, alpha, beta, gamma)
    """
    # pymatgen Lattice expects columns as vectors, we have rows
    lattice = Lattice(lattice_vectors.T)
    return np.array(lattice.parameters)

def compute_supercell_matrix_from_virtual_coords(
    prim_vectors: np.ndarray,
    supercell_vectors: np.ndarray
) -> np.ndarray:
    """
    Recover supercell matrix from primitive and supercell lattice vectors.

    Relation: supercell_vectors = prim_vectors @ supercell_matrix^T
    Solve for supercell_matrix.

    Args:
        prim_vectors: [3, 3] array of primitive lattice vectors (row vectors)
        supercell_vectors: [3, 3] array of supercell lattice vectors (row vectors)

    Returns:
        supercell_matrix: [3, 3] transformation matrix
    """
    # supercell_vectors^T = supercell_matrix @ prim_vectors^T
    # supercell_matrix = supercell_vectors^T @ (prim_vectors^T)^(-1)
    # supercell_matrix = supercell_vectors^T @ prim_vectors^(-T)
    supercell_matrix = np.linalg.solve(prim_vectors.T, supercell_vectors.T).T
    return supercell_matrix

def virtual_coords_to_lattice_and_supercell(
    prim_virtual_coords: torch.Tensor,
    supercell_virtual_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert virtual coordinates back to lattice parameters and supercell matrix.

    Args:
        prim_virtual_coords: [B, 3, 3] or [3, 3]
        supercell_virtual_coords: [B, 3, 3] or [3, 3]

    Returns:
        lattice_params: [B, 6] or [6]
        supercell_matrix: [B, 3, 3] or [3, 3]
    """
    # Handle both batched and single sample
    single_sample = (prim_virtual_coords.ndim == 2)
    if single_sample:
        prim_virtual_coords = prim_virtual_coords.unsqueeze(0)
        supercell_virtual_coords = supercell_virtual_coords.unsqueeze(0)

    B = prim_virtual_coords.shape[0]
    lattice_params = torch.zeros(B, 6)
    supercell_matrices = torch.zeros(B, 3, 3)

    for i in range(B):
        prim_vecs = prim_virtual_coords[i].cpu().numpy()
        supercell_vecs = supercell_virtual_coords[i].cpu().numpy()

        # Convert to lattice params
        params = lattice_vectors_to_params(prim_vecs)
        lattice_params[i] = torch.from_numpy(params)

        # Compute supercell matrix
        sm = compute_supercell_matrix_from_virtual_coords(prim_vecs, supercell_vecs)
        supercell_matrices[i] = torch.from_numpy(sm)

    if single_sample:
        return lattice_params[0], supercell_matrices[0]
    return lattice_params, supercell_matrices
```

### Normalization for Virtual Coords

Virtual coordinates need separate normalizers from atom coordinates since they have different scales and distributions.

```python
# Compute statistics from training data
def compute_virtual_coord_statistics(dataset):
    """Compute mean/std for primitive and supercell virtual coordinates."""
    prim_vecs = []
    supercell_vecs = []

    for sample in dataset:
        lattice = sample['lattice']  # [6]
        supercell_matrix = sample['supercell_matrix']  # [3, 3]
        prim_vec, supercell_vec = compute_virtual_coords(
            lattice.unsqueeze(0),
            supercell_matrix.unsqueeze(0)
        )
        prim_vecs.append(prim_vec.squeeze(0))
        supercell_vecs.append(supercell_vec.squeeze(0))

    prim_vecs = torch.stack(prim_vecs)  # [N, 3, 3]
    supercell_vecs = torch.stack(supercell_vecs)  # [N, 3, 3]

    prim_mean = prim_vecs.mean(dim=0)  # [3, 3]
    prim_std = prim_vecs.std(dim=0)    # [3, 3]
    supercell_mean = supercell_vecs.mean(dim=0)  # [3, 3]
    supercell_std = supercell_vecs.std(dim=0)    # [3, 3]

    return {
        'prim_virtual_mean': prim_mean,
        'prim_virtual_std': prim_std,
        'supercell_virtual_mean': supercell_mean,
        'supercell_virtual_std': supercell_std,
    }
```

---

## Architectural Changes Required

### 1. Data Structure Changes

#### LMDB Storage Format (`lmdb_dataset.py`)
**NO CHANGES REQUIRED** - Storage format remains the same:
```python
{
    "primitive_slab": ASE.Atoms,
    "supercell_matrix": np.array([3, 3]),
    "lattice": computed from primitive_slab.cell,
    "scaling_factor": float,
    ...
}
```

This minimizes data migration effort and maintains backward compatibility.

#### Batch Collation (`collate_fn_with_dynamic_padding`)
**Current batched tensors**:
```python
- prim_slab_cart_coords: [B, N_max, 3]
- ads_cart_coords: [B, M_max, 3]
- lattice: [B, 6]
- supercell_matrix: [B, 3, 3]
- scaling_factor: [B]
```

**Proposed batched tensors**:
```python
- prim_slab_cart_coords: [B, N_max, 3]
- ads_cart_coords: [B, M_max, 3]
- primitive_virtual_coords: [B, 3, 3]  # Computed from lattice
- supercell_virtual_coords: [B, 3, 3]  # Computed from lattice + supercell_matrix
- scaling_factor: [B]
```

**Changes**:
- Remove `lattice` and `supercell_matrix` from model inputs
- Add **conversion functions** to compute virtual coords:
  ```python
  def lattice_params_to_vectors(a, b, c, alpha, beta, gamma):
      """Convert lattice parameters to 3x3 matrix of lattice vectors (row vectors)"""
      # Using crystallographic conventions
      # Returns: [3, 3] array where each row is a lattice vector

  def compute_primitive_virtual_coords(lattice: [B, 6]) -> [B, 3, 3]:
      """Convert lattice params to primitive lattice vectors"""
      return lattice_params_to_vectors(lattice)

  def compute_supercell_virtual_coords(lattice: [B, 6], supercell_matrix: [B, 3, 3]) -> [B, 3, 3]:
      """Compute supercell lattice vectors"""
      prim_vectors = lattice_params_to_vectors(lattice)  # [B, 3, 3]
      # supercell_vectors = prim_vectors @ supercell_matrix.T
      return torch.bmm(prim_vectors, supercell_matrix.transpose(-2, -1))
  ```
- Apply conversions during collation, before passing to model
- No masking needed for virtual atoms (always 6 atoms present)

---

### 2. Model Architecture Changes

#### Encoder (`AtomAttentionEncoder` in `layers.py`)

**Current atom ordering in encoder**:
```
[prim_slab_0, ..., prim_slab_N-1, ads_0, ..., ads_M-1]
Total: N + M atoms
```

**Proposed atom ordering**:
```
[prim_slab_0, ..., prim_slab_N-1, ads_0, ..., ads_M-1,
 prim_virtual_0, prim_virtual_1, prim_virtual_2,
 supercell_virtual_0, supercell_virtual_1, supercell_virtual_2]
Total: N + M + 6 atoms
```

**Feature Engineering Changes**:

1. **Primitive Slab Features** (unchanged):
   ```
   [padding_mask(1), element_onehot(100)]
   Embedded via: Linear(101, atom_s)
   ```

2. **Adsorbate Features** (unchanged):
   ```
   [ref_pos(3), padding_mask(1), binding_onehot(100), element_onehot(100)]
   Embedded via: Linear(204, atom_s)
   ```

3. **Virtual Atom Features** (NEW):
   ```
   [virtual_id_onehot(6)]  # Which virtual atom (0-5)
   Embedded via: Linear(6, atom_s)
   ```

   Where virtual_id:
   - 0, 1, 2: Primitive lattice vectors (a, b, c)
   - 3, 4, 5: Supercell lattice vectors (a', b', c')

   Alternatively (more explicit):
   ```
   [padding_mask(1)=1, lattice_type_onehot(2), vector_id_onehot(3)]
   # lattice_type: 0=primitive, 1=supercell
   # vector_id: 0=first vector, 1=second, 2=third
   Embedded via: Linear(6, atom_s)
   ```

**Position Encoding Changes**:
- **Current**: Separate embeddings for coords, lattice, supercell_matrix
  ```python
  q = q + coord_embedding(coords) + lattice_embedding(lattice) +
      supercell_embedding(supercell_matrix) + scaling_embedding(scaling_factor)
  ```

- **Proposed**: Only coords and scaling (lattice info now in virtual atom coords)
  ```python
  q = q + coord_embedding(all_coords) + scaling_embedding(scaling_factor)
  # where all_coords = [prim_coords, ads_coords, prim_virtual_coords, supercell_virtual_coords]
  ```

**Masking Changes**:
- **Current masks**: `[prim_mask(N), ads_mask(M)]` → [B, N+M]
- **Proposed masks**: `[prim_mask(N), ads_mask(M), virtual_mask(6)]` → [B, N+M+6]
- Virtual mask is always `[True, True, True, True, True, True]` (no padding for virtual atoms)

#### Decoder (`AtomAttentionDecoder` in `layers.py`)

**Current output heads**:
```python
# Per-atom predictions
- feats_to_prim_slab_coords: [B, N, 3]  # From x_prim_slab features
- feats_to_ads_center: [B, 3]           # From global pooled x_ads - REMOVE
- feats_to_ads_rel_coords: [B, M, 3]    # From x_ads features - REPLACE

# Global predictions (from pooled prim_slab features)
- feats_to_lattice: [B, 6]              # REMOVE
- feats_to_supercell_matrix: [B, 3, 3]  # REMOVE
- feats_to_scaling_factor: [B]
```

**Proposed output heads**:
```python
# Per-atom predictions (uniform approach for all atom types)
- feats_to_prim_slab_coords: [B, N, 3]        # From x_prim_slab features
- feats_to_ads_coords: [B, M, 3]              # SIMPLIFIED: Direct prediction from x_ads features
- feats_to_prim_virtual_coords: [B, 3, 3]    # NEW: From x_prim_virtual features
- feats_to_supercell_virtual_coords: [B, 3, 3] # NEW: From x_supercell_virtual features

# Global predictions (from pooled prim_slab features)
- feats_to_scaling_factor: [B]
```

**Key Changes**:
1. Remove lattice and supercell_matrix heads (replaced by virtual atoms)
2. **SIMPLIFIED**: Remove adsorbate center + relative decomposition, predict directly
3. All atom coordinates now predicted uniformly (per-atom, from atom features)
4. Virtual atoms predicted per-atom (like other atoms), not via global pooling

**Output Processing**:
```python
# Split atom features
x_prim_slab = x[:, :N, :]                        # [B, N, hidden]
x_ads = x[:, N:N+M, :]                           # [B, M, hidden]
x_prim_virtual = x[:, N+M:N+M+3, :]              # [B, 3, hidden]
x_supercell_virtual = x[:, N+M+3:N+M+6, :]       # [B, 3, hidden]

# Coordinate predictions (uniform per-atom approach)
prim_coords = feats_to_prim_slab_coords(x_prim_slab)           # [B, N, 3]
ads_coords = feats_to_ads_coords(x_ads)                        # [B, M, 3] - SIMPLIFIED
prim_virtual_coords = feats_to_prim_virtual_coords(x_prim_virtual)  # [B, 3, 3]
supercell_virtual_coords = feats_to_supercell_virtual_coords(x_supercell_virtual)  # [B, 3, 3]

# Global predictions
scaling_factor = feats_to_scaling_factor(global_pool(x_prim_slab))  # [B]

# Apply bounding: tanh * 3.0 in normalized space (all coordinates)
```

---

### 3. Flow Matching Changes (`flow.py`)

#### Current Flow Variables
```python
Inputs: t, r_prim [B, N, 3], r_ads [B, M, 3], l [B, 6], sm [B, 3, 3], sf [B]
Model predicts: Δr_prim, Δr_ads, Δl, Δsm, Δsf
Integration: r_t = r_0 + ∫_0^t v dt for each component
```

#### Proposed Flow Variables
```python
Inputs: t, r_prim [B, N, 3], r_ads [B, M, 3],
        r_prim_virtual [B, 3, 3], r_supercell_virtual [B, 3, 3], sf [B]
Model predicts: Δr_prim, Δr_ads, Δr_prim_virtual, Δr_supercell_virtual, Δsf
Integration: r_t = r_0 + ∫_0^t v dt for each component (including virtual coords)
```

**Key Changes**:
1. Remove `l` and `sm` from flow inputs
2. Add `r_prim_virtual` and `r_supercell_virtual` as flow variables
3. **SIMPLIFIED**: Adsorbate coords treated same as primitive slab (direct flow, no center/relative decomposition)
4. All coordinate types follow identical flow dynamics
5. Remove velocity predictions `Δl` and `Δsm`
6. Add velocity predictions `Δr_prim_virtual` and `Δr_supercell_virtual`

#### Detailed Flow Logic
```python
# At time t, compute noisy versions (training) - uniform linear interpolation for all coords
r_prim_t = r_prim_1 * t + r_prim_0 * (1 - t)
r_ads_t = r_ads_1 * t + r_ads_0 * (1 - t)  # SIMPLIFIED: same as prim
r_prim_virtual_t = r_prim_virtual_1 * t + r_prim_virtual_0 * (1 - t)
r_supercell_virtual_t = r_supercell_virtual_1 * t + r_supercell_virtual_0 * (1 - t)
sf_t = sf_1 * t + sf_0 * (1 - t)

# Model predicts velocities (all per-atom, uniform approach)
v_prim, v_ads, v_prim_virtual, v_supercell_virtual, v_sf = model(
    t, r_prim_t, r_ads_t, r_prim_virtual_t, r_supercell_virtual_t, sf_t
)

# Compute flow matching loss (uniform MSE across all coordinate types)
loss = MSE(v_prim, r_prim_1 - r_prim_0) + MSE(v_ads, r_ads_1 - r_ads_0) +
       MSE(v_prim_virtual, r_prim_virtual_1 - r_prim_virtual_0) +
       MSE(v_supercell_virtual, r_supercell_virtual_1 - r_supercell_virtual_0) +
       MSE(v_sf, sf_1 - sf_0)
```

#### Sampling (Inference)
```python
# Start from prior
r_prim_0, r_ads_0, r_prim_virtual_0, r_supercell_virtual_0, sf_0 = sample_prior()

# Integrate forward using ODE solver (uniform approach)
def ode_func(t, state):
    r_prim, r_ads, r_prim_virtual, r_supercell_virtual, sf = state
    v_prim, v_ads, v_prim_virtual, v_supercell_virtual, v_sf = model(
        t, r_prim, r_ads, r_prim_virtual, r_supercell_virtual, sf
    )
    return [v_prim, v_ads, v_prim_virtual, v_supercell_virtual, v_sf]

# Solve ODE from t=0 to t=1
state_1 = odeint(ode_func, [r_prim_0, r_ads_0, r_prim_virtual_0, r_supercell_virtual_0, sf_0], [0, 1])
r_prim_1, r_ads_1, r_prim_virtual_1, r_supercell_virtual_1, sf_1 = state_1[-1]
```

---

### 4. Prior Sampling Changes (`prior.py`)

#### Prior Sampling Strategy
**Key Insight**: Keep existing prior sampling for lattice params and supercell matrix, then convert to virtual coords.

#### Current Prior Sampling (KEEP AS-IS)
```python
- prim_slab_coords_0 ~ N(0, σ²) in normalized space
- ads_coords_0 ~ N(0, σ²) in normalized space
- lattice_lengths_0 ~ LogNormal(loc, scale)
- lattice_angles_0 ~ Uniform(60°, 120°)
- supercell_matrix_0 ~ N(0, σ²) in normalized space
- scaling_factor_0 ~ N(0, σ²)
```

#### Proposed Prior Sampling (MINIMAL CHANGES)
```python
# Sample as before
- prim_slab_coords_0 ~ N(0, σ²) in normalized space
- ads_coords_0 ~ N(0, σ²) in normalized space
- lattice_lengths_0 ~ LogNormal(loc, scale)
- lattice_angles_0 ~ Uniform(60°, 120°)
- supercell_matrix_0 ~ N(0, σ²) in normalized space
- scaling_factor_0 ~ N(0, σ²)

# NEW: Convert to virtual coords for model input
- lattice_0 = concat(lattice_lengths_0, lattice_angles_0)  # [B, 6]
- prim_virtual_coords_0 = lattice_params_to_vectors(lattice_0)  # [B, 3, 3]
- supercell_virtual_coords_0 = prim_virtual_coords_0 @ supercell_matrix_0.T  # [B, 3, 3]
```

**Benefits of this approach**:
- ✅ Minimal code changes to prior sampling logic
- ✅ Keeps proven prior distributions (LogNormal for lengths, Uniform for angles)
- ✅ Guarantees valid lattices (lengths > 0, angles in valid range)
- ✅ No need to learn new normalizers for virtual coords in prior space
- ✅ Virtual coords are derived, not sampled directly

#### Normalizer Updates
**For training targets (data → model)**:
- Add `PrimVirtualCoordsNormalizer` with statistics from primitive lattice vectors
- Add `SupercellVirtualCoordsNormalizer` with statistics from supercell lattice vectors
- Compute mean/std of lattice vectors from training data
- Apply normalize/denormalize like other coordinates

**For prior sampling (prior → model)**:
- No normalizers needed for lattice params or supercell matrix (already exist)
- Virtual coords are computed from unnormalized lattice params, then normalized for model input

---

### 5. Assembly Changes (`assemble.py`)

#### Current Assembly
```python
assemble(
    generated_prim_slab_coords: [N, 3],
    generated_lattice: [6],  # (a, b, c, α, β, γ)
    generated_supercell_matrix: [3, 3],
    generated_scaling_factor: float,
    ads_coords: [M, 3],
    ...
) → ASE.Atoms
```

#### Proposed Assembly
```python
assemble(
    generated_prim_slab_coords: [N, 3],
    generated_prim_virtual_coords: [3, 3],      # Primitive lattice vectors
    generated_supercell_virtual_coords: [3, 3], # Supercell lattice vectors
    generated_scaling_factor: float,
    ads_coords: [M, 3],
    ...
) → ASE.Atoms
```

**Assembly Logic Changes**:

**Current flow**:
1. Create Structure from `lattice_params` + `prim_slab_coords`
2. Apply `supercell_matrix` transformation: `supercell = make_supercell(structure, supercell_matrix)`
3. Scale z-direction by `scaling_factor`
4. Add adsorbates

**Proposed flow** (clean reconstruction with 6 virtual atoms):
1. **Convert virtual coords back to lattice params and supercell matrix**:
   ```python
   # Extract lattice parameters from primitive virtual coords
   prim_lattice_matrix = prim_virtual_coords  # [3, 3] row vectors
   lattice_params = lattice_vectors_to_params(prim_lattice_matrix)  # [6]: (a, b, c, α, β, γ)

   # Compute supercell matrix from primitive and supercell vectors
   # supercell_vectors = prim_vectors @ supercell_matrix.T
   # supercell_matrix.T = prim_vectors^(-1) @ supercell_vectors
   # supercell_matrix = (prim_vectors^(-1) @ supercell_vectors).T
   supercell_matrix = torch.linalg.solve(prim_virtual_coords.T, supercell_virtual_coords.T).T
   ```

2. Create Structure from `lattice_params` + `prim_slab_coords` (same as before)

3. Apply `supercell_matrix` transformation: `supercell = make_supercell(structure, supercell_matrix)` (same as before)

4. Scale z-direction by `scaling_factor` (same as before)

5. Add adsorbates (same as before)

**Key Benefit**: With 6 virtual atoms, we can **perfectly reconstruct** the original assembly flow!

**Helper Functions** (NEW):
```python
def lattice_vectors_to_params(vectors: np.ndarray) -> np.ndarray:
    """Convert 3x3 lattice vectors (row vectors) to (a, b, c, α, β, γ)"""
    from pymatgen.core import Lattice
    lattice = Lattice(vectors)  # Assumes row vectors
    return np.array(lattice.parameters)

def compute_supercell_matrix_from_virtual_coords(
    prim_vectors: np.ndarray,
    supercell_vectors: np.ndarray
) -> np.ndarray:
    """
    Compute supercell matrix from primitive and supercell lattice vectors.

    Relation: supercell_vectors = prim_vectors @ supercell_matrix.T
    Solve for: supercell_matrix
    """
    # supercell_matrix.T = prim_vectors^(-1) @ supercell_vectors
    # supercell_matrix = (prim_vectors^(-1) @ supercell_vectors).T
    supercell_matrix_T = np.linalg.solve(prim_vectors.T, supercell_vectors.T)
    return supercell_matrix_T.T
```

---

### 6. Training Changes (`module/flow.py` - `FlowModule`)

#### Loss Computation Updates

**Current loss terms**:
```python
loss_prim_slab = MSE(pred_prim_coords, target_prim_coords)
loss_ads = MSE(pred_ads_coords, target_ads_coords)
loss_lattice = MSE(pred_lattice, target_lattice)
loss_supercell = MSE(pred_supercell_matrix, target_supercell_matrix)
loss_scaling = MSE(pred_scaling_factor, target_scaling_factor)
```

**Proposed loss terms**:
```python
loss_prim_slab = MSE(pred_prim_coords, target_prim_coords)
loss_ads = MSE(pred_ads_coords, target_ads_coords)
loss_virtual = MSE(pred_virtual_coords, target_virtual_coords)
loss_scaling = MSE(pred_scaling_factor, target_scaling_factor)
```

**Changes**:
- Remove `loss_lattice` and `loss_supercell`
- Add `loss_virtual` for virtual atom coordinates
- Virtual atoms contribute to total loss like regular atoms

#### Batch Expansion for Multiplicity
- Virtual atoms are expanded along with real atoms: [B, 3, 3] → [B*mult, 3, 3]
- Same multiplicity applied to all components

---

### 7. Validation Changes (`validation.py`)

#### Current Metrics
- RMSD: Align predicted and reference structures (real atoms only)
- Validity: Check if structure is chemically valid
- Energy prediction: Use predicted structure for ML potential

**Changes**:
- **RMSD**: Unchanged (still computed on real atoms only)
- **Virtual atom accuracy**: New metric for lattice vector prediction
  ```python
  virtual_mae = MAE(pred_virtual_coords, ref_virtual_coords)
  virtual_vector_angle_error = angle_between(pred_vectors, ref_vectors)
  virtual_vector_length_error = |length(pred) - length(ref)|
  ```
- **Lattice validity**: Check if virtual atoms form valid lattice
  ```python
  # Check if vectors are linearly independent
  det(virtual_coords.T) > threshold
  # Check if angles are reasonable
  angles_between_vectors in [60°, 120°]
  ```

#### Structure Reconstruction for Validation
- Need to convert virtual coords back to lattice + supercell matrix
- May require storing original supercell matrix as metadata

---

### 8. Data Generation Changes (`scripts/oc20_to_catgen.py`)

**NO CHANGES REQUIRED** - Data generation remains the same:

```python
# Extract from OC20 structure (current code, unchanged)
primitive_slab = structure[tags != -1]  # Real slab atoms
lattice_params = primitive_slab.cell.cellpar()
supercell_matrix = compute_supercell_matrix(structure)

# Store in LMDB (current format, unchanged)
{
    "primitive_slab": primitive_slab,
    "lattice": lattice_params,
    "supercell_matrix": supercell_matrix,
    ...
}
```

**Key Benefit**: No need to regenerate or migrate existing datasets! Virtual coords are computed on-the-fly during data loading.

---

## Implementation Challenges & Solutions

### Challenge 1: Virtual Atom Feature Design
**Problem**: Virtual atoms don't have element types or physical properties like real atoms.

**Solution**: Use position-based encoding (virtual_id = 0-5 one-hot encoding)
- IDs 0, 1, 2: Primitive lattice vectors (a, b, c)
- IDs 3, 4, 5: Supercell lattice vectors (a', b', c')

### Challenge 2: Conversion Between Representations
**Problem**: Need bidirectional conversion between (lattice_params, supercell_matrix) ↔ (prim_virtual_coords, supercell_virtual_coords).

**Solution**: Implement conversion functions
- Forward: `lattice_params_to_vectors()` and matrix multiplication
- Backward: `lattice_vectors_to_params()` and linear solve
- Use pymatgen for robust lattice parameter conversions

### Challenge 3: Prior Distribution for Virtual Atoms
**Problem**: Need to sample valid initial lattice vectors for flow matching.

**Solution**: Keep existing structured prior, then convert
- Sample lattice params (LogNormal for lengths, Uniform for angles)
- Sample supercell matrix (Gaussian in normalized space)
- Convert to virtual coords using conversion functions
- This guarantees valid lattices without new priors

### Challenge 4: Normalizer Statistics for Virtual Coords
**Problem**: Need to normalize/denormalize virtual coordinates for training.

**Solution**: Compute statistics from existing data
- Extract lattice vectors from all training samples
- Compute mean/std for primitive and supercell vectors separately
- Store in config files
- No need to modify LMDB data

### Challenge 5: Backwards Compatibility
**Problem**: Need to work with existing datasets.

**Solution**: No migration needed!
- LMDB format unchanged
- Conversion happens during batch collation
- Old checkpoints won't work (different architecture), but data is compatible
- Can train new models without regenerating data

---

## Final Architecture: 6 Virtual Atoms with Minimal Data Changes

### Final Design Decision
Use **6 virtual atoms** with **on-the-fly conversion** from existing data format:

1. **Primitive Cell Atoms** (N real atoms)
2. **Adsorbate Atoms** (M real atoms)
3. **Primitive Lattice Virtual Atoms** (3 virtual atoms)
   - Positions: [3, 3] representing primitive lattice vectors (a, b, c)
   - Computed from: lattice parameters during batch collation
4. **Supercell Lattice Virtual Atoms** (3 virtual atoms)
   - Positions: [3, 3] representing supercell lattice vectors (a', b', c')
   - Computed from: primitive vectors @ supercell_matrix.T during batch collation

### Key Benefits
✅ **Minimal data processing changes**: LMDB format unchanged, conversion during collation only
✅ **Complete representation**: Both primitive and supercell lattice explicit
✅ **Clean assembly**: Can reconstruct lattice params and supercell matrix perfectly
✅ **No data migration**: Works with existing datasets immediately
✅ **Valid priors**: Use existing structured priors (LogNormal, Uniform) then convert
✅ **Backward compatible data**: Only model architecture changes, not data format
✅ **Simplified architecture**: Uniform per-atom coordinate prediction for all atom types (no center/relative decomposition)

### Design Principles
1. **Data storage**: Keep lattice params + supercell matrix (unchanged)
2. **Data loading**: Convert to virtual coords during batch collation
3. **Model processing**: Work entirely in virtual atom coordinate space
4. **Assembly**: Convert virtual coords back to lattice params + supercell matrix
5. **Prior sampling**: Sample lattice params + supercell matrix, then convert
6. **Uniform coordinate prediction**: All atom coordinates (prim slab, adsorbate, virtual) predicted uniformly via per-atom heads (no special decomposition)

### Relationship Between Representations
```
Forward (data → model):
  lattice_params [6] → prim_virtual_coords [3, 3]
  prim_virtual_coords [3, 3] @ supercell_matrix.T [3, 3] → supercell_virtual_coords [3, 3]

Backward (model → assembly):
  prim_virtual_coords [3, 3] → lattice_params [6]
  solve(prim_virtual_coords, supercell_virtual_coords) → supercell_matrix [3, 3]
```

---

## Implementation Order

### Phase 1: Conversion Functions & Statistics
1. Implement `lattice_params_to_vectors()` function
2. Implement `lattice_vectors_to_params()` function
3. Implement `compute_supercell_matrix_from_virtual_coords()` function
4. Compute normalizer statistics for virtual atom coordinates from training data
5. Update config files with virtual coord normalizers

### Phase 2: Data Loading & Batch Collation
6. Update `collate_fn_with_dynamic_padding()` to compute virtual coords
7. Modify batch dict to include `primitive_virtual_coords` and `supercell_virtual_coords`
8. Remove `lattice` and `supercell_matrix` from model inputs
9. Test data loading pipeline with new tensors

### Phase 3: Model Architecture
10. Update encoder to handle 6 virtual atoms (feature engineering)
11. Update decoder to predict virtual atom coordinates (2 new heads)
12. Remove `feats_to_lattice` and `feats_to_supercell_matrix` heads
13. Update atom concatenation and masking for N+M+6 atoms
14. Update token aggregation if needed

### Phase 4: Training Infrastructure
15. Update flow module to include virtual coords in flow variables
16. Update prior sampler to convert lattice params → virtual coords
17. Update loss computation (replace lattice/supercell losses with virtual losses)
18. Update weight tensors and config for new architecture

### Phase 5: Inference and Assembly
19. Update assembly to convert virtual coords → lattice params + supercell matrix
20. Update sampling scripts to handle virtual atoms
21. Update validation metrics for virtual atom accuracy
22. Test end-to-end generation pipeline

### Phase 6: Testing and Validation
23. Unit tests for conversion functions
24. Integration tests for data loading
25. Forward pass tests with virtual atoms
26. Assembly tests for reconstruction accuracy
27. End-to-end sampling tests

---

## Files to Modify

### Critical Path (Must Change)
1. `src/catgen/data/lmdb_dataset.py` - Batch collation with virtual coord conversion
2. `src/catgen/models/layers.py` - Encoder/decoder with virtual atom heads
3. `src/catgen/module/flow.py` - Flow matching with virtual atoms
4. `src/catgen/data/prior.py` - Prior sampling with virtual coord conversion
5. `src/catgen/scripts/assemble.py` - Assembly with virtual coord → lattice conversion

### Supporting Files (Need Updates)
6. `src/catgen/validation.py` - Validation metrics for virtual atoms
7. `src/catgen/scripts/sample.py` - Sampling logic
8. `configs/*` - Config files for virtual coord normalizer statistics

### New Files (To Create)
9. `src/catgen/data/conversions.py` - Conversion functions for lattice ↔ virtual coords
10. `src/scripts/compute_virtual_atom_stats.py` - Compute normalizer statistics

### Files NOT Modified (Benefit of minimal changes)
- `src/scripts/oc20_to_catgen.py` - NO CHANGES (data format unchanged)
- LMDB datasets - NO CHANGES (no migration needed)

---

## Testing Strategy

### Unit Tests
- [ ] Virtual atom feature engineering
- [ ] Virtual atom coordinate prediction
- [ ] Prior sampling for virtual atoms
- [ ] Normalizer for virtual coords
- [ ] Assembly with virtual atoms

### Integration Tests
- [ ] End-to-end data loading with virtual atoms
- [ ] Forward pass with virtual atoms
- [ ] Loss computation with virtual atoms
- [ ] Sampling and assembly pipeline

### Validation Tests
- [ ] Compare RMSD with old architecture
- [ ] Check lattice validity from virtual atoms
- [ ] Verify supercell transformation accuracy
- [ ] Test on held-out structures

---

## Resolved Design Decisions

1. **Number of virtual atoms**: ✅ 6 atoms (3 primitive + 3 supercell)
2. **Data format**: ✅ Keep existing LMDB format, convert on-the-fly
3. **Virtual atom features**: ✅ One-hot encoding of virtual_id (0-5)
4. **Prior sampling**: ✅ Sample lattice params + supercell matrix, then convert
5. **Assembly**: ✅ Convert virtual coords back to lattice params + supercell matrix
6. **Attention mechanism**: ✅ Treat virtual atoms like regular atoms
7. **Position encoding**: ✅ Yes, apply to all atoms including virtual
8. **Adsorbate coordinate prediction**: ✅ Direct per-atom prediction (no center/relative decomposition)

## Remaining Open Questions

1. **Should we enforce relationship between primitive and supercell virtual coords during training?**
   - Option A: Let model learn freely (may violate supercell = prim @ matrix relationship)
   - Option B: Add constraint loss: `||supercell_pred - prim_pred @ supercell_matrix||²`
   - Option C: Hard constraint by predicting only prim + matrix, derive supercell
   - **Recommendation**: Start with Option A (unconstrained), add Option B if needed

2. **How to initialize virtual atom prediction heads?**
   - Option A: Random initialization
   - Option B: Initialize to predict mean virtual coords from training data
   - **Recommendation**: Option A for simplicity

3. **Should virtual coords be normalized differently from atom coords?**
   - Option A: Use separate normalizers (different statistics)
   - Option B: Use same normalizer as atom coords
   - **Recommendation**: Option A (separate normalizers) for better scaling

4. **How to handle degenerate lattices in predictions?**
   - Option A: Post-process to project onto valid lattice space
   - Option B: Trust model to learn valid lattices
   - Option C: Add validity loss term
   - **Recommendation**: Start with Option B, add validation checks during inference

---

## Success Criteria

- [ ] Model trains without errors
- [ ] Loss converges similar to original architecture
- [ ] Generated structures have valid lattices (det > 0, reasonable angles)
- [ ] RMSD comparable to original architecture
- [ ] Virtual atoms accurately represent lattice vectors (MAE < 0.1 Å)
- [ ] Assembly produces valid ASE.Atoms objects
- [ ] Sampling generates diverse structures

---

## Timeline Estimate

With the simplified approach (no data migration):
- Conversion functions & statistics: ~0.5 days
- Data loading & collation: ~1 day
- Model architecture: ~2 days
- Training infrastructure: ~1.5 days
- Inference and assembly: ~1.5 days
- Testing and debugging: ~2 days
- **Total: ~8-9 days (1.5 weeks)**

Simplified from original estimate due to:
- ✅ No LMDB format changes
- ✅ No data migration scripts
- ✅ Existing prior distributions reused
- ✅ Simpler conversion logic

---

## Next Steps

### Immediate Actions
1. ✅ **Design decision**: Confirmed 6 virtual atoms with on-the-fly conversion
2. **Compute statistics**: Run script to compute virtual coord normalizer stats
3. **Implement conversions**: Create `src/catgen/data/conversions.py`
4. **Update data loading**: Modify `collate_fn_with_dynamic_padding()`

### Phase-by-Phase Plan
Follow the implementation order in Phase 1 → Phase 6 above.

### Testing Strategy
- Unit test conversion functions (bidirectional, accuracy)
- Test data loading with virtual atoms
- Test model forward pass
- Test assembly reconstruction accuracy
- End-to-end generation test

---

## Summary of Key Design Choices

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Number of virtual atoms | 6 (3 prim + 3 supercell) | Complete representation, clean assembly |
| Data storage format | Unchanged (lattice + matrix) | No migration, backward compatible |
| Conversion timing | On-the-fly during collation | Minimal code changes |
| Prior sampling | Existing + conversion | Proven distributions, valid lattices |
| Virtual atom features | One-hot ID (0-5) | Simple, distinguishes prim vs supercell |
| Assembly | Convert back to lattice + matrix | Reuses existing logic |
| Coordinate prediction | Uniform per-atom for all types | Simplified, no center/relative decomposition |

---

**Document Status**: Finalized plan
**Last Updated**: 2026-01-12
**Author**: Claude (AI Assistant)
**Approved**: Based on user requirements
