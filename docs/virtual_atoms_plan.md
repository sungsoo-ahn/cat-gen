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

### Design Clarification Needed

**Interpretation A** (3 virtual atoms total):
- 3 virtual atoms represent the **supercell lattice vectors**
- Replaces both `lattice` (6 params) and `supercell_matrix` (9 params)
- Each virtual atom position = one supercell lattice vector (a', b', c')
- **Trade-off**: Loses explicit primitive cell lattice representation

**Interpretation B** (6 virtual atoms total):
- 3 virtual atoms for **primitive lattice vectors** (a, b, c)
- 3 virtual atoms for **supercell lattice vectors** (a', b', c')
- More explicit but contradicts "three additional supercell atoms"

**Recommended: Interpretation A** (based on "three additional supercell atoms")

### Proposed Atom Types
1. **Primitive Cell Atoms** (N real atoms)
   - Type: Real atoms with atomic numbers 1-100
   - Positions: [N, 3] in Cartesian coordinates
   - Features: Element one-hot + padding mask

2. **Adsorbate Atoms** (M real atoms)
   - Type: Real atoms with atomic numbers 1-100
   - Positions: [M, 3] in Cartesian coordinates
   - Features: Element one-hot + ref position + binding atom + padding mask

3. **Supercell Virtual Atoms** (3 virtual atoms)
   - Type: Virtual atoms with special atomic number (e.g., 0 or 101)
   - Positions: [3, 3] where each row is a supercell lattice vector
   - Features: Special embedding for virtual atom type
   - Represents: The 3 supercell lattice vectors (a', b', c')
   - **No padding**: Always present (no variable length)

---

## Architectural Changes Required

### 1. Data Structure Changes

#### LMDB Storage Format (`lmdb_dataset.py`)
**Current**:
```python
{
    "primitive_slab": ASE.Atoms,
    "supercell_matrix": np.array([3, 3]),
    "lattice": computed from primitive_slab,
    "scaling_factor": float,
    ...
}
```

**Proposed**:
```python
{
    "primitive_slab": ASE.Atoms,
    "supercell_lattice_vectors": np.array([3, 3]),  # Virtual atom positions
    "scaling_factor": float,
    ...
}
```

**Changes**:
- Compute supercell lattice vectors: `supercell_vectors = lattice_matrix @ supercell_matrix.T`
- Store as `supercell_lattice_vectors` [3, 3] where each row is a vector
- Remove separate `lattice` and `supercell_matrix` fields

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
- supercell_virtual_coords: [B, 3, 3]  # No padding needed (always 3)
- scaling_factor: [B]
```

**Changes**:
- Add `supercell_virtual_coords` to batch
- Remove `lattice` and `supercell_matrix` from batch
- No masking needed for virtual atoms (always present)

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
[prim_slab_0, ..., prim_slab_N-1, ads_0, ..., ads_M-1, virtual_0, virtual_1, virtual_2]
Total: N + M + 3 atoms
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
   [virtual_type_onehot(3)]  # Which virtual atom (0, 1, or 2)
   Embedded via: Linear(3, atom_s)
   ```

   Alternatively:
   ```
   [padding_mask(1)=1, virtual_indicator(1)=1, virtual_id(3)=onehot]
   Embedded via: Linear(5, atom_s)
   ```

**Position Encoding Changes**:
- **Current**: Separate embeddings for coords, lattice, supercell_matrix
  ```python
  q = q + coord_embedding(coords) + lattice_embedding(lattice) +
      supercell_embedding(supercell_matrix) + scaling_embedding(scaling_factor)
  ```

- **Proposed**: Only coords and scaling (lattice info in virtual atom coords)
  ```python
  q = q + coord_embedding(all_coords) + scaling_embedding(scaling_factor)
  # where all_coords = [prim_coords, ads_coords, virtual_coords]
  ```

**Masking Changes**:
- **Current masks**: `[prim_mask(N), ads_mask(M)]` → [B, N+M]
- **Proposed masks**: `[prim_mask(N), ads_mask(M), virtual_mask(3)]` → [B, N+M+3]
- Virtual mask is always `[True, True, True]` (no padding)

#### Decoder (`AtomAttentionDecoder` in `layers.py`)

**Current output heads** (from global pooling of prim_slab):
```python
- feats_to_prim_slab_coords: [B, N, 3]
- feats_to_ads_center: [B, 3]
- feats_to_ads_rel_coords: [B, M, 3]
- feats_to_lattice: [B, 6]  # REMOVE
- feats_to_supercell_matrix: [B, 3, 3]  # REMOVE
- feats_to_scaling_factor: [B]
```

**Proposed output heads**:
```python
- feats_to_prim_slab_coords: [B, N, 3]
- feats_to_ads_center: [B, 3]
- feats_to_ads_rel_coords: [B, M, 3]
- feats_to_virtual_coords: [B, 3, 3]  # NEW: Direct prediction of virtual atom positions
- feats_to_scaling_factor: [B]
```

**Key Change**: Virtual atoms are predicted like regular atoms (per-atom prediction), not via global pooling.

**Output Processing**:
- Virtual atoms split from main tensor: `x_virtual = x[:, N+M:N+M+3, :]` → [B, 3, hidden]
- Virtual coords: `feats_to_virtual_coords(x_virtual)` → [B, 3, 3]
- Apply same bounding: `tanh * 3.0` in normalized space

---

### 3. Flow Matching Changes (`flow.py`)

#### Current Flow Variables
```python
t, r_prim, r_ads, l, sm, sf → model → (Δr_prim, Δr_ads, Δl, Δsm, Δsf)
```

#### Proposed Flow Variables
```python
t, r_prim, r_ads, r_virtual, sf → model → (Δr_prim, Δr_ads, Δr_virtual, Δsf)
```

**Changes**:
- Remove `l_noisy` and `sm_noisy` from flow inputs
- Add `r_virtual_noisy` [B, 3, 3] to flow inputs
- Remove `Δl` and `Δsm` from outputs
- Add `Δr_virtual` [B, 3, 3] to outputs

#### Integration Updates
```python
# Current
r_t = r_0 + ∫_0^t v_r dt
l_t = l_0 + ∫_0^t v_l dt
sm_t = sm_0 + ∫_0^t v_sm dt

# Proposed
r_t = r_0 + ∫_0^t v_r dt  (includes virtual coords)
```

Virtual atoms follow same flow dynamics as real atoms.

---

### 4. Prior Sampling Changes (`prior.py`)

#### Current Prior Sampling
```python
- prim_slab_coords_0 ~ N(0, σ²) in normalized space
- ads_coords_0 ~ N(0, σ²) in normalized space
- lattice_lengths_0 ~ LogNormal(loc, scale)  # REMOVE
- lattice_angles_0 ~ Uniform(60°, 120°)  # REMOVE
- supercell_matrix_0 ~ N(0, σ²) in normalized space  # REMOVE
- scaling_factor_0 ~ N(0, σ²)
```

#### Proposed Prior Sampling
```python
- prim_slab_coords_0 ~ N(0, σ²) in normalized space
- ads_coords_0 ~ N(0, σ²) in normalized space
- virtual_coords_0 ~ ??? (see options below)
- scaling_factor_0 ~ N(0, σ²)
```

**Options for Virtual Atom Prior**:

**Option 1**: Gaussian in normalized space (like other coords)
```python
virtual_coords_0 ~ N(0, σ²) in normalized space → denormalize → [B, 3, 3]
```
- Pros: Consistent with atom coord priors
- Cons: May sample unrealistic lattices (collapsed, inverted)

**Option 2**: Gaussian in raw space with statistics from data
```python
virtual_coords_0 ~ N(μ_virtual, Σ_virtual) where μ, Σ from training data
```
- Pros: More realistic initial lattices
- Cons: Less flexibility, need to compute statistics

**Option 3**: Structured prior enforcing lattice properties
```python
# Sample lengths and angles, then convert to vectors
lengths ~ LogNormal(...)
angles ~ Uniform(60°, 120°)
virtual_coords = lattice_params_to_vectors(lengths, angles)
```
- Pros: Guarantees valid lattices
- Cons: More complex, less end-to-end

**Recommendation**: Start with **Option 1** for simplicity and consistency.

#### Normalizer Updates
- Add `VirtualCoordsNormalizer` with statistics from supercell lattice vectors
- Compute mean/std of `[a', b', c']` vectors from training data
- Apply same normalize/denormalize as other coordinates

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
    generated_virtual_coords: [3, 3],  # Supercell lattice vectors
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

**Proposed flow**:
1. **Virtual coords → Supercell lattice matrix**: `supercell_lattice = virtual_coords.T` (3x3 matrix with columns = vectors)
2. Create supercell directly: `supercell = Structure(lattice=supercell_lattice, species=..., coords=...)`
3. **Problem**: How to place primitive cell atoms in the supercell?

**Challenge**: Without explicit primitive lattice and supercell transformation, we need to:
- **Option A**: Assume primitive coords are in fractional coordinates of an implicit primitive cell
- **Option B**: Treat primitive coords as Cartesian, tile them to fill supercell
- **Option C**: Reconstruct primitive lattice from virtual atoms (requires 6 virtual atoms)

**Recommended Solution**:
- Store primitive cell atoms in **fractional coordinates** in the primitive cell
- During assembly:
  1. Reconstruct primitive lattice by "undoing" the supercell transformation
  2. Supercell transformation is unknown, but we can estimate it from generated structure
  3. Or: Store supercell matrix separately as metadata (not learned)

**Alternative Design** (may require rethinking):
- Use **6 virtual atoms**: 3 for primitive lattice + 3 for supercell lattice
- This makes assembly straightforward but contradicts "three additional atoms"

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

### 8. Data Generation Changes (`scripts/oc20_to_mincatflow.py`)

**Current conversion**:
```python
# Extract from OC20 structure
primitive_slab = structure[tags != -1]  # Real slab atoms
lattice_params = primitive_slab.cell.cellpar()
supercell_matrix = compute_supercell_matrix(structure)
```

**Proposed conversion**:
```python
# Extract supercell lattice vectors
supercell_cell = structure.cell  # Full slab cell
supercell_lattice_vectors = supercell_cell.array  # [3, 3] row vectors

# Convert to virtual atom positions
virtual_coords = supercell_lattice_vectors  # Each row = vector
```

**Key consideration**:
- Need to handle the primitive cell → supercell transformation
- May need to store primitive lattice separately for reconstruction

---

## Implementation Challenges & Solutions

### Challenge 1: Loss of Primitive Lattice Information
**Problem**: With only 3 virtual atoms for supercell, we lose explicit primitive cell lattice.

**Solutions**:
1. **Store primitive lattice as metadata** (not learned)
2. **Use 6 virtual atoms** (3 primitive + 3 supercell)
3. **Encode primitive info in atom features** (e.g., fractional coordinates)

**Recommendation**: Option 2 (6 virtual atoms) for complete representation.

### Challenge 2: Virtual Atom Feature Design
**Problem**: Virtual atoms don't have element types or physical properties.

**Solutions**:
1. Use special "virtual" element type (atomic number 0 or 101)
2. Use position-based encoding (virtual_id = 0, 1, 2 or 0, 1, 2, 3, 4, 5)
3. Use learnable embeddings for virtual atom types

**Recommendation**: Option 2 with one-hot encoding of virtual atom ID.

### Challenge 3: Prior Distribution for Virtual Atoms
**Problem**: Virtual atoms represent lattice vectors, need valid lattice constraints.

**Solutions**:
1. Unconstrained Gaussian (may sample invalid lattices)
2. Structured prior (lengths + angles → vectors)
3. Data-driven statistics (mean + covariance from training data)

**Recommendation**: Start with Option 1, add constraints if needed.

### Challenge 4: Assembly Without Primitive Lattice
**Problem**: Can't reconstruct full structure without knowing primitive cell.

**Solutions**:
1. Store supercell matrix as metadata
2. Use 6 virtual atoms (primitive + supercell)
3. Assume primitive coords are fractional and estimate primitive lattice

**Recommendation**: Option 2 (6 virtual atoms) for clean reconstruction.

### Challenge 5: Backwards Compatibility
**Problem**: Existing trained models and datasets use old format.

**Solutions**:
1. Migrate all datasets to new format (conversion script)
2. Support both formats with version flag
3. Train new models from scratch

**Recommendation**: Option 1 + Option 3 (clean break, train new models).

---

## Revised Recommendation: 6 Virtual Atoms

Based on the challenges above, I recommend using **6 virtual atoms** instead of 3:

### Revised Architecture
1. **Primitive Cell Atoms** (N real atoms)
2. **Adsorbate Atoms** (M real atoms)
3. **Primitive Lattice Virtual Atoms** (3 virtual atoms)
   - Positions: [3, 3] representing primitive lattice vectors (a, b, c)
4. **Supercell Lattice Virtual Atoms** (3 virtual atoms)
   - Positions: [3, 3] representing supercell lattice vectors (a', b', c')

### Benefits
- Complete representation: both primitive and supercell explicit
- Clean assembly: can reconstruct structure unambiguously
- Preserves all current information (lattice + supercell matrix)
- Easier to validate (check both primitive and supercell lattices)

### Trade-offs
- More virtual atoms (6 instead of 3)
- Slightly more complex model
- Need to enforce relationship: `supercell_vectors = primitive_vectors @ supercell_matrix.T`

---

## Implementation Order

### Phase 1: Data Preparation
1. Update LMDB dataset format to store virtual atom coordinates
2. Write conversion script from old format to new format
3. Compute statistics for virtual atom normalizers
4. Test data loading with new format

### Phase 2: Model Architecture
5. Update encoder to handle virtual atoms (feature engineering)
6. Update decoder to predict virtual atom coordinates
7. Remove lattice/supercell matrix heads
8. Update token aggregation for N+M+6 atoms

### Phase 3: Training Infrastructure
9. Update flow module to include virtual coords in flow
10. Update prior sampler for virtual atoms
11. Update loss computation
12. Update batch collation and padding

### Phase 4: Inference and Validation
13. Update assembly to use virtual atoms
14. Update validation metrics
15. Update sampling scripts
16. Test end-to-end generation

---

## Files to Modify

### Critical Path (Must Change)
1. `src/catgen/data/lmdb_dataset.py` - Data loading, collation, virtual atoms
2. `src/catgen/models/layers.py` - Encoder/decoder with virtual atom heads
3. `src/catgen/module/flow.py` - Flow matching with virtual atoms
4. `src/catgen/data/prior.py` - Prior sampling for virtual atoms
5. `src/catgen/scripts/assemble.py` - Assembly from virtual atoms

### Supporting Files (Need Updates)
6. `src/scripts/oc20_to_mincatflow.py` - Data conversion
7. `src/catgen/validation.py` - Validation metrics
8. `src/catgen/scripts/sample.py` - Sampling logic
9. `configs/*` - Config files for normalizer statistics

### New Files (To Create)
10. `scripts/data_generation/convert_to_virtual_atoms.sh` - Migration script
11. `src/scripts/compute_virtual_atom_stats.py` - Compute normalizer stats

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

## Open Questions

1. **Should we use 3 or 6 virtual atoms?**
   - 3 atoms: Only supercell (loses primitive lattice)
   - 6 atoms: Both primitive and supercell (complete but more complex)
   - **Recommendation**: 6 atoms for completeness

2. **How to enforce lattice validity?**
   - Unconstrained: May generate invalid lattices
   - Constrained: Project to valid lattice space
   - **Recommendation**: Start unconstrained, add projection if needed

3. **How to handle virtual atoms in attention?**
   - Treat like regular atoms (attend to each other)
   - Special attention mask (virtual-virtual, virtual-real)
   - **Recommendation**: Treat like regular atoms for simplicity

4. **What atomic number for virtual atoms?**
   - 0: Natural choice for "no element"
   - 101: Beyond real elements
   - **Recommendation**: 0 with special handling

5. **Should virtual atoms have position encoding?**
   - Yes: Consistent with real atoms
   - No: Already represent positions
   - **Recommendation**: Yes, for consistency

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

- Data preparation: ~2 days
- Model architecture: ~3 days
- Training infrastructure: ~2 days
- Inference and validation: ~2 days
- Testing and debugging: ~3 days
- **Total: ~2 weeks**

(Note: Actual time may vary based on implementation challenges)

---

## Next Steps

1. **Clarify design decision**: 3 vs 6 virtual atoms
2. **Review this plan** with team/advisor
3. **Create detailed implementation checklist**
4. **Set up development branch** for this feature
5. **Begin Phase 1**: Data preparation and conversion

---

**Document Status**: Draft for review
**Last Updated**: 2026-01-12
**Author**: Claude (AI Assistant)
