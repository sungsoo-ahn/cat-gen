"""Centralized constants for the CatGen codebase.

This module contains all magic numbers and constants used throughout the codebase.
Import from here instead of defining locally to ensure consistency.
"""

# =============================================================================
# Numerical Stability
# =============================================================================

# Standard epsilon for numerical stability in divisions
EPS_NUMERICAL = 1e-8

# Flow matching epsilon (used in ODE solver and flow computations)
FLOW_EPSILON = 1e-5

# =============================================================================
# Element Encoding
# =============================================================================

# Number of elements for one-hot encoding (covers most elements in periodic table)
# Elements 1-100 (H to Fm)
NUM_ELEMENTS = 100

# Discrete Flow Matching constants (for dng=True mode)
# Mask token is index 0, actual elements are 1-100
MASK_TOKEN_INDEX = 0
NUM_ELEMENTS_WITH_MASK = NUM_ELEMENTS + 1  # 101 total (0=MASK, 1-100=elements)

# =============================================================================
# Loss Scaling
# =============================================================================

# Expected variance of angles in degrees^2 (for normalizing angle loss)
ANGLE_LOSS_SCALE = 300.0

# =============================================================================
# Data Handling
# =============================================================================

# Sentinel value for missing reference energy in LMDB records
MISSING_REF_ENERGY = float("nan")

# Default padding value for atomic numbers (0 = mask/pad)
PAD_VALUE = 0
