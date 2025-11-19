#!/usr/bin/env python3
"""
test_pdb_conversion.py
---------------------
Test script to verify PDB-to-realistic-map conversion works correctly.

Usage:
    python test_pdb_conversion.py <input.pdb>
"""

import os
import sys
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from pdb_to_realistic_map import PDBToRealisticMap
from Bio.PDB import PDBParser


def analyze_pdb(pdb_file: str):
    """Analyze original PDB structure."""
    print("\n" + "="*70)
    print("ORIGINAL PDB ANALYSIS")
    print("="*70)
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # Count atoms
    atoms = list(structure.get_atoms())
    n_atoms = len(atoms)
    
    # Get coordinates
    coords = np.array([atom.get_coord() for atom in atoms])
    
    # Compute bounds
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    size = maxs - mins
    center = coords.mean(axis=0)
    
    print(f"File: {os.path.basename(pdb_file)}")
    print(f"Number of atoms: {n_atoms:,}")
    print(f"Size (Å): X={size[0]:.1f}, Y={size[1]:.1f}, Z={size[2]:.1f}")
    print(f"Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    
    return {
        'n_atoms': n_atoms,
        'size': size,
        'center': center,
        'coords': coords
    }


def analyze_converted_map(map_file: str):
    """Analyze converted cryo-EM map."""
    print("\n" + "="*70)
    print("CONVERTED MAP ANALYSIS")
    print("="*70)
    
    with mrcfile.open(map_file, permissive=True) as mrc:
        data = mrc.data
        
        # Get voxel size
        try:
            vs = mrc.voxel_size
            if hasattr(vs, 'x'):
                voxel_size = float(np.mean([vs.x, vs.y, vs.z]))
            else:
                voxel_size = float(np.mean(vs))
        except:
            voxel_size = 1.0
        
        # Map statistics
        shape = data.shape
        density_mean = data.mean()
        density_std = data.std()
        density_min = data.min()
        density_max = data.max()
        
        # Non-zero voxels
        threshold = density_mean + 0.5 * density_std
        n_nonzero = (data > threshold).sum()
        
        print(f"File: {os.path.basename(map_file)}")
        print(f"Shape: {shape}")
        print(f"Voxel size: {voxel_size:.3f} Å")
        print(f"Physical size: {shape[0]*voxel_size:.1f} × {shape[1]*voxel_size:.1f} × {shape[2]*voxel_size:.1f} Å")
        print(f"\nDensity statistics:")
        print(f"  Mean: {density_mean:.6f}")
        print(f"  Std:  {density_std:.6f}")
        print(f"  Min:  {density_min:.6f}")
        print(f"  Max:  {density_max:.6f}")
        print(f"\nVoxels above threshold ({threshold:.3f}): {n_nonzero:,} ({100*n_nonzero/data.size:.2f}%)")
        
        return {
            'shape': shape,
            'voxel_size': voxel_size,
            'data': data,
            'n_nonzero': n_nonzero
        }


def check_virion_separation(map_file: str):
    """Check if virions are properly separated."""
    print("\n" + "="*70)
    print("VIRION SEPARATION CHECK")
    print("="*70)
    
    with mrcfile.open(map_file, permissive=True) as mrc:
        data = mrc.data
        
        # Threshold at mean + 2*std
        threshold = data.mean() + 2 * data.std()
        binary = data > threshold
        
        # Find connected components (simplified - project to 2D)
        from scipy import ndimage
        
        # Project to XY plane
        projection = binary.max(axis=0)
        
        # Label connected components
        labeled, n_components = ndimage.label(projection)
        
        print(f"Detected {n_components} separate density regions in XY projection")
        
        if n_components >= 3:
            print("✓ At least 3 virions detected (GOOD)")
            
            # Measure separation
            centers = []
            for i in range(1, min(4, n_components+1)):
                mask = (labeled == i)
                if mask.any():
                    coords = np.argwhere(mask)
                    center = coords.mean(axis=0)
                    centers.append(center)
            
            if len(centers) >= 2:
                # Compute distances
                min_dist = float('inf')
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        min_dist = min(min_dist, dist)
                
                # Convert to Angstroms
                try:
                    vs = mrc.voxel_size
                    if hasattr(vs, 'x'):
                        voxel_size = float(np.mean([vs.x, vs.y, vs.z]))
                    else:
                        voxel_size = float(np.mean(vs))
                except:
                    voxel_size = 1.0
                
                min_dist_angstrom = min_dist * voxel_size
                print(f"Minimum separation: {min_dist:.1f} voxels = {min_dist_angstrom:.1f} Å")
                
                if min_dist_angstrom >= 150:
                    print("✓ Separation ≥ 150 Å (EXCELLENT)")
                elif min_dist_angstrom >= 100:
                    print("⚠ Separation < 150 Å but > 100 Å (ACCEPTABLE)")
                else:
                    print("✗ Separation < 100 Å (TOO CLOSE)")
        else:
            print(f"⚠ Only {n_components} regions detected (expected 3)")
            print("  This might affect clustering quality")


def visualize_map(map_file: str, output_prefix: str = None):
    """Create visualization of the converted map."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    with mrcfile.open(map_file, permissive=True) as mrc:
        data = mrc.data
    
    # Create figure with multiple views
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Converted Map: {os.path.basename(map_file)}', fontsize=16)
    
    # Get central slices
    z_mid = data.shape[0] // 2
    y_mid = data.shape[1] // 2
    x_mid = data.shape[2] // 2
    
    # XY slices (at different Z)
    axes[0, 0].imshow(data[z_mid-20], cmap='gray')
    axes[0, 0].set_title(f'XY Slice (Z={z_mid-20})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(data[z_mid], cmap='gray')
    axes[0, 1].set_title(f'XY Slice (Z={z_mid}, center)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(data[z_mid+20], cmap='gray')
    axes[0, 2].set_title(f'XY Slice (Z={z_mid+20})')
    axes[0, 2].axis('off')
    
    # Orthogonal views
    axes[1, 0].imshow(data[:, y_mid, :], cmap='gray')
    axes[1, 0].set_title(f'XZ Slice (Y={y_mid})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(data[:, :, x_mid], cmap='gray')
    axes[1, 1].set_title(f'YZ Slice (X={x_mid})')
    axes[1, 1].axis('off')
    
    # Histogram
    axes[1, 2].hist(data.flatten(), bins=100, alpha=0.7)
    axes[1, 2].axvline(data.mean(), color='r', linestyle='--', label='Mean')
    axes[1, 2].axvline(data.mean() + 2*data.std(), color='g', linestyle='--', label='Mean+2σ')
    axes[1, 2].set_xlabel('Density Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Density Distribution')
    axes[1, 2].legend()
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    
    # Save or show
    if output_prefix:
        output_file = f"{output_prefix}_visualization.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {output_file}")
    else:
        plt.show()
    
    plt.close()


def run_test(pdb_file: str, output_dir: str = None):
    """Run complete test suite."""
    
    if not os.path.exists(pdb_file):
        print(f"Error: File not found: {pdb_file}")
        return False
    
    print("\n" + "#"*70)
    print("PDB-TO-MAP CONVERSION TEST")
    print("#"*70)
    print(f"Input: {pdb_file}")
    
    # Analyze original PDB
    pdb_info = analyze_pdb(pdb_file)
    
    # Convert to map
    print("\n" + "="*70)
    print("CONVERTING PDB TO REALISTIC MAP")
    print("="*70)
    
    converter = PDBToRealisticMap()
    
    # Set output location
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_map = os.path.join(output_dir, 'test_converted.mrc')
    else:
        output_map = 'test_converted.mrc'
    
    try:
        result_map = converter.convert(
            pdb_file,
            output_file=output_map,
            apix=1.2,
            resolution=6.0,
            box_size=256,
            n_virions=3,
            min_separation=150.0,
            noise_level=2.0
        )
        print(f"✓ Conversion successful: {result_map}")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze converted map
    map_info = analyze_converted_map(result_map)
    
    # Check virion separation
    check_virion_separation(result_map)
    
    # Visualize
    output_prefix = output_map.replace('.mrc', '') if output_dir else None
    visualize_map(result_map, output_prefix)
    
    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ PDB atoms:          {pdb_info['n_atoms']:,}")
    print(f"✓ Map voxels:         {np.prod(map_info['shape']):,}")
    print(f"✓ Non-zero voxels:    {map_info['n_nonzero']:,}")
    print(f"✓ Voxel size:         {map_info['voxel_size']:.3f} Å")
    print(f"✓ Physical size:      {map_info['shape'][0]*map_info['voxel_size']:.1f} Å")
    
    ratio = map_info['n_nonzero'] / pdb_info['n_atoms']
    print(f"\nVoxel/Atom ratio:     {ratio:.1f}x")
    
    if ratio > 5:
        print("✓ Good voxel sampling (PASS)")
    else:
        print("⚠ Low voxel sampling (consider smaller apix)")
    
    print("\n" + "#"*70)
    print("TEST COMPLETE")
    print("#"*70)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pdb_conversion.py <input.pdb> [output_dir]")
        print("\nExample:")
        print("  python test_pdb_conversion.py 1abc_HA.pdb ./test_output/")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = run_test(pdb_file, output_dir)
    
    sys.exit(0 if success else 1)