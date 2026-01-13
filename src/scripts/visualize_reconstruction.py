#!/usr/bin/env python3
"""Visualize the OC20 to CatGen reconstruction process.

Shows step-by-step:
1. Original slab (from OC20 metadata)
2. Tight slab (vacuum removed)
3. Primitive cell
4. Reconstructed slab
5. Final system with adsorbate

Usage:
    uv run python src/scripts/visualize_reconstruction.py --index 0 --split train
"""

import argparse
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure, Lattice
from src.helpers import calculate_rmsd_pymatgen


def parse_value(val, dtype=None, np_type=None):
    """Parse string values from CSV."""
    if isinstance(val, str):
        val = ast.literal_eval(val)
    if dtype == 'array':
        return np.array(val, dtype=np_type)
    elif dtype:
        return dtype(val)
    return val


def plot_atoms_3d(ax, atoms, title, color_map=None, alpha=0.8):
    """Plot ASE Atoms on a 3D axis."""
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    
    # Default color map by atomic number
    if color_map is None:
        color_map = {
            1: 'white', 6: 'gray', 7: 'blue', 8: 'red',
            13: 'pink', 45: 'silver', 46: 'lightblue',
            47: 'silver', 72: 'cyan', 50: 'brown'
        }
    
    # Get colors for each atom
    colors = [color_map.get(n, 'green') for n in numbers]
    
    # Plot atoms
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=colors, s=100, alpha=alpha, edgecolors='black', linewidth=0.5)
    
    # Plot cell
    cell = atoms.get_cell()
    origin = np.array([0, 0, 0])
    
    # Draw cell edges
    for i in range(3):
        ax.plot3D(*zip(origin, cell[i]), 'k-', alpha=0.3)
    
    for i in range(3):
        for j in range(i+1, 3):
            ax.plot3D(*zip(cell[i], cell[i] + cell[j]), 'k-', alpha=0.3)
            ax.plot3D(*zip(cell[j], cell[i] + cell[j]), 'k-', alpha=0.3)
    
    # Draw remaining edges
    ax.plot3D(*zip(cell[0] + cell[1], cell[0] + cell[1] + cell[2]), 'k-', alpha=0.3)
    ax.plot3D(*zip(cell[0] + cell[2], cell[0] + cell[1] + cell[2]), 'k-', alpha=0.3)
    ax.plot3D(*zip(cell[1] + cell[2], cell[0] + cell[1] + cell[2]), 'k-', alpha=0.3)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)


def visualize_reconstruction(df_row, output_path=None):
    """Visualize the full reconstruction process."""
    
    # Parse data from row
    n_c = parse_value(df_row['n_c'], int)
    n_vac = parse_value(df_row['n_vac'], int)
    sid = parse_value(df_row['sid'], int)
    tags = parse_value(df_row['true_tags'], 'array')
    atomic_numbers = parse_value(df_row['true_system_atomic_numbers'], 'array', int)
    positions = parse_value(df_row['true_system_positions'], 'array', float)
    lattice = parse_value(df_row['true_lattice'], 'array', float)
    ads_pos_relaxed = parse_value(df_row['ads_pos_relaxed'], 'array', float)
    
    print(f"Processing sample SID: {sid}")
    print(f"  n_slab_layers: {n_c}, n_vacuum_layers: {n_vac}")
    print(f"  Total atoms: {len(atomic_numbers)}")
    print(f"  Slab atoms: {(tags != 2).sum()}, Adsorbate atoms: {(tags == 2).sum()}")
    
    # Create original system
    true_system = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=lattice,
        tags=tags,
        pbc=True
    )
    
    # Separate slab and adsorbate
    slab_mask = (tags == 0) | (tags == 1)
    true_slab = true_system[slab_mask]
    adsorbate = true_system[tags == 2]
    
    adaptor = AseAtomsAdaptor()
    n_layers = n_c + n_vac
    
    # Step 1: Create tight slab (remove vacuum)
    tight_slab = true_slab.copy()
    tight_cell = tight_slab.get_cell()
    tight_cell[2] = tight_cell[2] * (n_c / n_layers)
    tight_slab.set_cell(tight_cell)
    tight_slab.center()
    
    print(f"\nStep 1: Tight slab (vacuum removed)")
    print(f"  Original cell[2] length: {np.linalg.norm(true_slab.cell[2]):.2f} Å")
    print(f"  Tight cell[2] length: {np.linalg.norm(tight_slab.cell[2]):.2f} Å")
    
    # Step 2: Find primitive cell
    tight_slab_struct = adaptor.get_structure(tight_slab)
    prim_slab_struct = tight_slab_struct.get_primitive_structure(tolerance=0.1)
    
    print(f"\nStep 2: Primitive cell")
    print(f"  Tight slab atoms: {len(tight_slab)}")
    print(f"  Primitive cell atoms: {len(prim_slab_struct)}")
    print(f"  Reduction factor: {len(tight_slab) / len(prim_slab_struct):.1f}x")
    
    # Step 3: Standardize orientation
    a, b, c, alpha, beta, gamma = prim_slab_struct.lattice.parameters
    standard_lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    
    L_orig = prim_slab_struct.lattice.matrix
    L_std = standard_lattice.matrix
    R = (np.linalg.inv(L_orig) @ L_std).T
    
    rotated_prim_struct = Structure(
        standard_lattice,
        prim_slab_struct.species,
        prim_slab_struct.frac_coords
    )
    
    # Step 4: Compute supercell matrix
    rotated_tight_lattice = Lattice(tight_slab_struct.lattice.matrix @ R.T)
    sc_matrix = np.round(np.dot(
        rotated_tight_lattice.matrix,
        np.linalg.inv(standard_lattice.matrix)
    )).astype(int)
    
    print(f"\nStep 3: Supercell matrix")
    print(f"  {sc_matrix[0]}")
    print(f"  {sc_matrix[1]}")
    print(f"  {sc_matrix[2]}")
    
    # Step 5: Reconstruct
    recon_struct = rotated_prim_struct.copy()
    recon_struct.make_supercell(sc_matrix, to_unit_cell=False)
    recon_tight_slab = adaptor.get_atoms(recon_struct)
    
    # Verify reconstruction of tight slab
    rmsd_tight, _ = calculate_rmsd_pymatgen(
        struct1=recon_tight_slab,
        struct2=tight_slab,
        ltol=0.2, stol=0.3, angle_tol=5,
        primitive_cell=False,
    )
    
    print(f"\nStep 4: Reconstruction verification")
    print(f"  RMSD (tight slab): {rmsd_tight:.6f} Å")
    
    # Step 6: Add vacuum back
    scaling_factor = n_layers / n_c
    recon_slab = recon_tight_slab.copy()
    recon_cell = recon_slab.get_cell()
    recon_cell[2] = recon_cell[2] * scaling_factor
    recon_slab.set_cell(recon_cell)
    
    rmsd_w_vac, _ = calculate_rmsd_pymatgen(
        struct1=recon_slab,
        struct2=true_slab,
        ltol=0.2, stol=0.3, angle_tol=5,
        primitive_cell=False,
    )
    print(f"  RMSD (with vacuum): {rmsd_w_vac:.6f} Å")
    
    # Step 7: Transform adsorbate
    rotated_true_slab = true_slab.copy()
    rotated_true_slab.set_cell(true_slab.cell @ R.T)
    rotated_true_slab.set_positions(true_slab.positions @ R.T)
    
    rotated_adsorbate = adsorbate.copy()
    rotated_adsorbate.set_positions(adsorbate.positions @ R.T)
    diff_vector = recon_slab.get_center_of_mass() - rotated_true_slab.get_center_of_mass()
    rotated_adsorbate.translate(diff_vector)
    
    # Final system
    final_system = recon_slab + rotated_adsorbate
    
    rmsd_system, _ = calculate_rmsd_pymatgen(
        struct1=final_system,
        struct2=true_system,
        ltol=0.2, stol=0.3, angle_tol=5,
        primitive_cell=False,
    )
    print(f"  RMSD (full system): {rmsd_system:.6f} Å")
    
    # Get primitive slab as ASE Atoms
    primitive_slab_atoms = adaptor.get_atoms(rotated_prim_struct)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Original system
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_atoms_3d(ax1, true_system, f'1. Original System\n({len(true_system)} atoms)')
    
    # Plot 2: Original slab only
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_atoms_3d(ax2, true_slab, f'2. Original Slab\n({len(true_slab)} atoms)')
    
    # Plot 3: Tight slab
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    plot_atoms_3d(ax3, tight_slab, f'3. Tight Slab (no vacuum)\n({len(tight_slab)} atoms)')
    
    # Plot 4: Primitive cell
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    plot_atoms_3d(ax4, primitive_slab_atoms, f'4. Primitive Cell\n({len(primitive_slab_atoms)} atoms)')
    
    # Plot 5: Reconstructed slab
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    plot_atoms_3d(ax5, recon_slab, f'5. Reconstructed Slab\n({len(recon_slab)} atoms)')
    
    # Plot 6: Final system with adsorbate
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    plot_atoms_3d(ax6, final_system, f'6. Reconstructed + Adsorbate\n({len(final_system)} atoms)')
    
    plt.suptitle(f'Reconstruction Process - SID: {sid}\n'
                 f'Supercell: {sc_matrix[0].tolist()}, {sc_matrix[1].tolist()}, {sc_matrix[2].tolist()}\n'
                 f'RMSD: tight={rmsd_tight:.2e}, vacuum={rmsd_w_vac:.2e}, system={rmsd_system:.2e}',
                 fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure to: {output_path}")
    else:
        plt.savefig('reconstruction_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved figure to: reconstruction_visualization.png")
    
    plt.show()
    
    return {
        'sid': sid,
        'n_slab': n_c,
        'n_vac': n_vac,
        'n_atoms_original': len(true_slab),
        'n_atoms_primitive': len(primitive_slab_atoms),
        'supercell_matrix': sc_matrix,
        'rmsd_tight': rmsd_tight,
        'rmsd_vacuum': rmsd_w_vac,
        'rmsd_system': rmsd_system,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize reconstruction process")
    parser.add_argument('--index', type=int, default=0, help='Sample index in metadata')
    parser.add_argument('--split', type=str, default='train', help='Dataset split')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    args = parser.parse_args()
    
    metadata_path = f"metadata/{args.split}_metadata.csv"
    print(f"Loading metadata from: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    print(f"Total samples: {len(df)}")
    
    if args.index >= len(df):
        print(f"Error: Index {args.index} out of range (max: {len(df)-1})")
        return
    
    row = df.iloc[args.index]
    result = visualize_reconstruction(row, args.output)
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Original slab atoms: {result['n_atoms_original']}")
    print(f"  Primitive cell atoms: {result['n_atoms_primitive']}")
    print(f"  Compression ratio: {result['n_atoms_original'] / result['n_atoms_primitive']:.1f}x")
    print(f"  All RMSD < 1e-4: {all(r < 1e-4 for r in [result['rmsd_tight'], result['rmsd_vacuum'], result['rmsd_system']])}")


if __name__ == "__main__":
    main()
