"""Test script to verify supercell matrix and virtual atom math is correct."""

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

# Import the conversion functions
import sys
sys.path.insert(0, '/home/user/cat-gen')
from src.catgen.data.conversions import (
    lattice_params_to_vectors,
    lattice_vectors_to_params,
    compute_virtual_coords,
    virtual_coords_to_lattice_and_supercell,
    compute_supercell_matrix_from_vectors,
)


def test_forward_backward_consistency():
    """Test that we can recover lattice params and supercell matrix from virtual coords."""
    print("=" * 80)
    print("TEST 1: Forward-Backward Consistency")
    print("=" * 80)

    # Create test lattice parameters
    lattice_params = torch.tensor([[10.0, 10.0, 15.0, 90.0, 90.0, 90.0]], dtype=torch.float32)

    # Create test supercell matrix (2x2x1 supercell)
    supercell_matrix = torch.tensor([[[2.0, 0.0, 0.0],
                                       [0.0, 2.0, 0.0],
                                       [0.0, 0.0, 1.0]]], dtype=torch.float32)

    print(f"Input lattice params: {lattice_params}")
    print(f"Input supercell matrix:\n{supercell_matrix[0]}")

    # Forward: lattice_params + supercell_matrix -> virtual coords
    prim_virtual, supercell_virtual = compute_virtual_coords(lattice_params, supercell_matrix)

    print(f"\nPrimitive virtual coords (lattice vectors as rows):\n{prim_virtual[0]}")
    print(f"\nSupercell virtual coords:\n{supercell_virtual[0]}")

    # Backward: virtual coords -> lattice_params + supercell_matrix
    recovered_lattice, recovered_supercell = virtual_coords_to_lattice_and_supercell(
        prim_virtual, supercell_virtual
    )

    print(f"\nRecovered lattice params: {recovered_lattice}")
    print(f"Recovered supercell matrix:\n{recovered_supercell[0]}")

    # Check errors
    lattice_error = torch.abs(lattice_params - recovered_lattice).max().item()
    supercell_error = torch.abs(supercell_matrix - recovered_supercell).max().item()

    print(f"\nMax lattice error: {lattice_error:.2e}")
    print(f"Max supercell matrix error: {supercell_error:.2e}")

    if lattice_error < 1e-4 and supercell_error < 1e-4:
        print("✓ PASS: Forward-backward consistency is good!")
    else:
        print("✗ FAIL: Errors are too large!")

    return lattice_error < 1e-4 and supercell_error < 1e-4


def test_pymatgen_compatibility():
    """Test that our supercell transformation matches pymatgen's behavior."""
    print("\n" + "=" * 80)
    print("TEST 2: Pymatgen Compatibility")
    print("=" * 80)

    # Create a primitive structure
    lattice = Lattice.from_parameters(a=5.0, b=5.0, c=10.0, alpha=90, beta=90, gamma=90)
    prim_struct = Structure(lattice, ["Cu", "Cu"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    print(f"Primitive lattice matrix (rows = [a, b, c]):\n{prim_struct.lattice.matrix}")

    # Define supercell matrix
    supercell_matrix_np = np.array([[2, 0, 0],
                                     [0, 2, 0],
                                     [0, 0, 1]])

    print(f"\nSupercell matrix:\n{supercell_matrix_np}")

    # Method 1: Use pymatgen's make_supercell
    sc_struct = prim_struct.copy()
    sc_struct.make_supercell(supercell_matrix_np)
    pymatgen_sc_lattice = sc_struct.lattice.matrix

    print(f"\nPymatgen supercell lattice:\n{pymatgen_sc_lattice}")

    # Method 2: Use our conversion functions
    prim_lattice_torch = torch.from_numpy(prim_struct.lattice.matrix).float().unsqueeze(0)
    supercell_matrix_torch = torch.from_numpy(supercell_matrix_np).float().unsqueeze(0)

    # Compute supercell lattice using our formula
    our_sc_lattice = torch.bmm(supercell_matrix_torch, prim_lattice_torch)[0].numpy()

    print(f"\nOur supercell lattice (S @ P):\n{our_sc_lattice}")

    # Compare
    error = np.abs(pymatgen_sc_lattice - our_sc_lattice).max()
    print(f"\nMax difference: {error:.2e}")

    if error < 1e-6:
        print("✓ PASS: Our formula matches pymatgen!")
    else:
        print("✗ FAIL: Formula mismatch!")

    return error < 1e-6


def test_matrix_shapes_and_row_order():
    """Test that matrix shapes and row ordering are correct."""
    print("\n" + "=" * 80)
    print("TEST 3: Matrix Shapes and Row Order")
    print("=" * 80)

    # Create a non-square lattice to test row order
    lattice_params = torch.tensor([[5.0, 8.0, 12.0, 90.0, 90.0, 90.0]], dtype=torch.float32)

    # Convert to vectors
    prim_vectors = lattice_params_to_vectors(lattice_params)

    print(f"Lattice params: a={lattice_params[0, 0]}, b={lattice_params[0, 1]}, c={lattice_params[0, 2]}")
    print(f"\nPrimitive vectors (as rows):\n{prim_vectors[0]}")

    # Check that row 0 has length ≈ a
    row0_len = torch.norm(prim_vectors[0, 0, :]).item()
    row1_len = torch.norm(prim_vectors[0, 1, :]).item()
    row2_len = torch.norm(prim_vectors[0, 2, :]).item()

    print(f"\n||row 0|| = {row0_len:.6f} (should be ≈ {lattice_params[0, 0]:.6f})")
    print(f"||row 1|| = {row1_len:.6f} (should be ≈ {lattice_params[0, 1]:.6f})")
    print(f"||row 2|| = {row2_len:.6f} (should be ≈ {lattice_params[0, 2]:.6f})")

    error_a = abs(row0_len - lattice_params[0, 0].item())
    error_b = abs(row1_len - lattice_params[0, 1].item())
    error_c = abs(row2_len - lattice_params[0, 2].item())

    if error_a < 1e-4 and error_b < 1e-4 and error_c < 1e-4:
        print("✓ PASS: Row order is correct!")
        return True
    else:
        print("✗ FAIL: Row order mismatch!")
        return False


def test_determinant_preservation():
    """Test that det(supercell_vectors) = det(supercell_matrix) * det(prim_vectors)."""
    print("\n" + "=" * 80)
    print("TEST 4: Determinant Preservation")
    print("=" * 80)

    lattice_params = torch.tensor([[10.0, 10.0, 15.0, 90.0, 90.0, 120.0]], dtype=torch.float32)
    supercell_matrix = torch.tensor([[[2.0, 1.0, 0.0],
                                       [0.0, 2.0, 0.0],
                                       [0.0, 0.0, 3.0]]], dtype=torch.float32)

    prim_virtual, supercell_virtual = compute_virtual_coords(lattice_params, supercell_matrix)

    det_prim = torch.det(prim_virtual[0]).item()
    det_supercell = torch.det(supercell_virtual[0]).item()
    det_matrix = torch.det(supercell_matrix[0]).item()

    print(f"det(prim_vectors) = {det_prim:.6f}")
    print(f"det(supercell_matrix) = {det_matrix:.6f}")
    print(f"det(supercell_vectors) = {det_supercell:.6f}")
    print(f"det(S) * det(P) = {det_matrix * det_prim:.6f}")

    error = abs(det_supercell - det_matrix * det_prim)
    print(f"\nError: {error:.2e}")

    if error < 1e-3:
        print("✓ PASS: Determinant relation holds!")
        return True
    else:
        print("✗ FAIL: Determinant mismatch!")
        return False


def test_ill_conditioned_matrix():
    """Test numerical stability with ill-conditioned matrices."""
    print("\n" + "=" * 80)
    print("TEST 5: Numerical Stability with Ill-Conditioned Matrices")
    print("=" * 80)

    # Create an ill-conditioned lattice (nearly degenerate)
    lattice_params = torch.tensor([[10.0, 10.0, 0.01, 90.0, 90.0, 90.0]], dtype=torch.float32)
    supercell_matrix = torch.tensor([[[2.0, 0.0, 0.0],
                                       [0.0, 2.0, 0.0],
                                       [0.0, 0.0, 1.0]]], dtype=torch.float32)

    print(f"Ill-conditioned lattice params (c=0.01): {lattice_params}")

    try:
        prim_virtual, supercell_virtual = compute_virtual_coords(lattice_params, supercell_matrix)
        recovered_lattice, recovered_supercell = virtual_coords_to_lattice_and_supercell(
            prim_virtual, supercell_virtual
        )

        lattice_error = torch.abs(lattice_params - recovered_lattice).max().item()
        supercell_error = torch.abs(supercell_matrix - recovered_supercell).max().item()

        print(f"\nMax lattice error: {lattice_error:.2e}")
        print(f"Max supercell error: {supercell_error:.2e}")

        if lattice_error < 1e-2 and supercell_error < 1e-2:
            print("✓ PASS: Handles ill-conditioned matrices reasonably")
            return True
        else:
            print("⚠ WARNING: Large errors with ill-conditioned matrix")
            return False
    except Exception as e:
        print(f"✗ FAIL: Exception raised: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SUPERCELL MATRIX AND VIRTUAL ATOM CORRECTNESS TESTS")
    print("=" * 80)

    results = []
    results.append(("Forward-Backward Consistency", test_forward_backward_consistency()))
    results.append(("Pymatgen Compatibility", test_pymatgen_compatibility()))
    results.append(("Matrix Shapes and Row Order", test_matrix_shapes_and_row_order()))
    results.append(("Determinant Preservation", test_determinant_preservation()))
    results.append(("Numerical Stability", test_ill_conditioned_matrix()))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 80)
