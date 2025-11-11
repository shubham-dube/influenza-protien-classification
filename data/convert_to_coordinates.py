#!/usr/bin/env python3
"""
convert_map_to_coords.py
------------------------

Convert a cryo-EM density map (.map or .mrc) into coordinate formats (.csv, .xyz).

Usage:
    python convert_map_to_coords.py input_file.map
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import mrcfile
except ImportError:
    print("❌ Missing dependency: Please install mrcfile using:")
    print("   pip install mrcfile pandas numpy")
    sys.exit(1)


# -------------------------------------------------------------------------
# Function: process_map
# -------------------------------------------------------------------------
def process_map(input_file: str, threshold: float = 0.1, downsample: int = 4) -> pd.DataFrame:
    """
    Extract 3D coordinates from a .map or .mrc cryo-EM density file.

    Parameters
    ----------
    input_file : str
        Path to the .map or .mrc file.
    threshold : float, optional
        Density threshold; only voxels with values above this will be kept.
    downsample : int, optional
        Keep every Nth voxel to reduce output size (default = 4).

    Returns
    -------
    pd.DataFrame
        DataFrame containing 3D coordinates (x, y, z).
    """
    print(f"[INFO] Reading density map: {input_file}")

    # Open file safely
    with mrcfile.open(input_file, permissive=True) as mrc:
        data = mrc.data.copy()

        # Compute voxel size safely
        try:
            if hasattr(mrc, "voxel_size"):
                vs = mrc.voxel_size
                if hasattr(vs, "x") and hasattr(vs, "y") and hasattr(vs, "z"):
                    voxel_size = float(np.mean([vs.x, vs.y, vs.z]))
                else:
                    voxel_size = float(np.mean(vs))
            else:
                voxel_size = 1.0
        except Exception:
            voxel_size = 1.0

    print(f"[INFO] Applying threshold: {threshold}, downsampling: {downsample}")
    print(f"[INFO] Using voxel size: {voxel_size:.4f}")

    # Get coordinates of density > threshold
    coords = np.argwhere(data > threshold)

    # Reduce number of points
    coords = coords[::downsample]

    # Scale coordinates by voxel size
    coords = coords * voxel_size

    # Convert to DataFrame
    df = pd.DataFrame(coords, columns=["x", "y", "z"])
    print(f"[INFO] Extracted {len(df)} coordinates.")
    return df


# -------------------------------------------------------------------------
# Function: save_outputs
# -------------------------------------------------------------------------
def save_outputs(df: pd.DataFrame, output_base: str) -> None:
    """
    Save extracted coordinates to .csv and .xyz files.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing (x, y, z) coordinates.
    output_base : str
        Base filename for saving output.
    """
    csv_path = f"{output_base}.csv"
    xyz_path = f"{output_base}.xyz"

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV file: {csv_path}")

    # Save XYZ
    with open(xyz_path, "w") as f:
        f.write(f"{len(df)}\nGenerated from density map\n")
        for _, row in df.iterrows():
            f.write(f"C {row['x']:.3f} {row['y']:.3f} {row['z']:.3f}\n")
    print(f"✅ Saved XYZ file: {xyz_path}")


# -------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_map_to_coords.py <input_file.map>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not input_file.endswith((".map", ".mrc")):
        print("❌ Please provide a .map or .mrc file.")
        sys.exit(1)

    output_base = os.path.splitext(input_file)[0]
    df = process_map(input_file)
    save_outputs(df, output_base)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
