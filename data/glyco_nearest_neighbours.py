#!/usr/bin/env python3
"""
Compute nearest-neighbour distance matrix from cleaned coordinates
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Compute nearest-neighbour distances")
    parser.add_argument("glycos", type=str, help="CSV file of cleaned coordinates (from previous step)")
    parser.add_argument("--prefix", type=str, default="test", help="Prefix for output files")
    parser.add_argument("-d", "--datadir", type=str, default=os.getcwd(), help="Output directory")
    parser.add_argument("-p", "--pixel", type=float, default=0.209, help="Pixel size (nm per unit)")
    parser.add_argument("-k", "--kth", type=int, default=1, help="K-th neighbour to ignore (usually 1)")
    return parser.parse_args()


def main():
    args = parse_args()
    datadir, glycos, prefix, pixel, K_th = args.datadir, args.glycos, args.prefix, args.pixel, args.kth

    print(f"[INFO] Reading {glycos}")
    vectors = np.loadtxt(glycos, delimiter=",", skiprows=1)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    K_corr = K_th + 3
    nbrs = NearestNeighbors(n_neighbors=K_corr, algorithm="brute").fit(vectors)
    results, indices = nbrs.kneighbors(vectors)

    # Remove self-distance (0) and apply pixel scaling
    dist = np.delete(results, np.s_[:K_th:], 1)
    dist = np.multiply(dist, pixel)

    dist_df = pd.DataFrame(dist)
    out_path = os.path.join(datadir, f"{prefix}_nbr.csv")
    dist_df.to_csv(out_path, index=False)
    print(f"[INFO] Nearest-neighbour distances saved → {out_path}")
# python3 glyco_nearest_neighbours.py ./output/HA_clean_glyco.csv --prefix HA --datadir ./output --pixel 0.209

if __name__ == "__main__":
    main()
