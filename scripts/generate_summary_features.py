#!/usr/bin/env python3
"""
Generate summary-level features per virion for HA/NA.
Computes mean_dist, std_dist, aspect_ratio, num_points, class_label.
"""

import os
import pandas as pd
import numpy as np
import open3d as o3d

def summarize_protein(prefix, datadir, class_label):
    """Summarize all *_glyco.csv and *_mesh.stl in a directory."""
    features = []
    for i in range(3):   # you have 3 clusters per protein
        glyco_path = os.path.join(datadir, f"{prefix}_clean_{i}_glyco.csv")
        mesh_path  = os.path.join(datadir, f"{prefix}_clean_{i}_mesh.stl")
        nbr_path   = os.path.join(datadir, f"{prefix}_nbr.csv")

        if not os.path.exists(glyco_path) or not os.path.exists(nbr_path):
            continue

        # --- Point statistics ---
        df_points = pd.read_csv(glyco_path)
        num_points = len(df_points)

        # --- Neighbour statistics ---
        df_nbr = pd.read_csv(nbr_path)
        mean_dist = df_nbr.values.mean()
        std_dist  = df_nbr.values.std()

        # --- Mesh-based aspect ratio ---
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        bbox = mesh.get_axis_aligned_bounding_box()
        extents = bbox.get_extent()
        aspect_ratio = max(extents) / min(extents) if min(extents) > 0 else 0

        features.append([mean_dist, std_dist, aspect_ratio, num_points, class_label])

    return features


if __name__ == "__main__":
    base = "data"
    out_path = "training_data/training_data_summary.csv"

    all_features = []
    all_features += summarize_protein("HA", f"{base}/HA_data_items", class_label=0)
    all_features += summarize_protein("NA", f"{base}/NA_data_items", class_label=1)

    df = pd.DataFrame(all_features, columns=[
        "mean_dist", "std_dist", "aspect_ratio", "num_points", "class_label"
    ])
    os.makedirs("training_data", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved summary dataset → {out_path}")
