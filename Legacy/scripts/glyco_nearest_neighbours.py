#!/usr/bin/env python3
"""
glyco_nearest_neighbours.py
---------------------------

Compute nearest-neighbour distance matrix from cleaned glycoprotein coordinates.

Usage:
    python3 glyco_nearest_neighbours.py ./output/HA_clean_glyco.csv \
        --prefix HA --datadir ./output --pixel 0.209
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# -------------------------------------------------------------------------
# Argument parser
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Compute nearest-neighbour distances from 3D coordinates.")
    parser.add_argument("glycos", type=str, help="CSV file of cleaned coordinates (from previous step)")
    parser.add_argument("--prefix", type=str, default="test", help="Prefix for output files")
    parser.add_argument("-d", "--datadir", type=str, default=os.getcwd(), help="Output directory")
    parser.add_argument("-p", "--pixel", type=float, default=0.209, help="Pixel size (nm per unit)")
    parser.add_argument("-k", "--kth", type=int, default=1, help="K-th neighbour to ignore (usually 1)")
    return parser.parse_args()


# -------------------------------------------------------------------------
# Utility: Load coordinates safely
# -------------------------------------------------------------------------
def load_coordinates(filepath: str) -> np.ndarray:
    """
    Load 3D coordinates from a CSV file, ignoring extra columns or headers.

    Supports both:
      - Files with headers (x,y,z)
      - Files without headers (plain numbers)
    """
    try:
        df = pd.read_csv(filepath)
        # Try to detect coordinate columns
        possible_cols = [c for c in df.columns if c.lower() in ["x", "y", "z"]]
        if len(possible_cols) >= 3:
            coords = df[possible_cols[:3]].to_numpy(dtype=float)
        else:
            # fallback: assume numeric only
            coords = df.to_numpy(dtype=float)
        return coords
    except Exception as e:
        print(f"[ERROR] Failed to load coordinates: {e}")
        return np.array([])


# -------------------------------------------------------------------------
# Main logic
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    datadir, glycos, prefix, pixel, K_th = args.datadir, args.glycos, args.prefix, args.pixel, args.kth

    os.makedirs(datadir, exist_ok=True)

    print(f"[INFO] Reading coordinates from: {glycos}")
    vectors = load_coordinates(glycos)
    if vectors.size == 0:
        print("❌ No valid coordinates found. Exiting.")
        return

    n_points = len(vectors)
    print(f"[INFO] Loaded {n_points} points for nearest-neighbour computation.")

    # Adjust number of neighbours to avoid exceeding dataset size
    K_corr = min(K_th + 3, n_points)
    if K_corr <= 1:
        print("❌ Not enough points to compute neighbours.")
        return

    print(f"[INFO] Computing distances with K={K_corr} (ignoring {K_th} self neighbours)...")

    # Fit nearest neighbours model
    nbrs = NearestNeighbors(n_neighbors=K_corr, algorithm="brute").fit(vectors)
    results, indices = nbrs.kneighbors(vectors)

    # Remove first K_th self-distances (usually 1 → self)
    dist = np.delete(results, np.s_[:K_th:], axis=1)

    # Apply pixel scaling
    dist_scaled = dist * pixel

    # Save results
    out_path = os.path.join(datadir, f"{prefix}_nbr.csv")
    pd.DataFrame(dist_scaled).to_csv(out_path, index=False)

    print(f"[INFO] Nearest-neighbour distances saved → {out_path}")
    print(f"[INFO] Shape of distance matrix: {dist_scaled.shape}")


# Usage :
#  python3 scripts/glyco_nearest_neighbours.py ./data/NA_data_items/NA_clean_glyco.csv --prefix NA --datadir ./data/NA_data_items --pixel 0.209
#  python3 scripts/glyco_nearest_neighbours.py ./data/HA_data_items/HA_clean_glyco.csv --prefix HA --datadir ./data/HA_data_items --pixel 0.209

# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()