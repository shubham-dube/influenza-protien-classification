#!/usr/bin/env python3
"""
Influenza morphological cleaner and reconstructor
Phase-1 : point-cloud cleaning, clustering and STL mesh generation
"""

import os
import argparse
import open3d as o3d
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and cluster 3D point-cloud data of influenza proteins.")
    parser.add_argument("coordinates", type=str, help="Input .xyz coordinate file")
    parser.add_argument("--prefix", type=str, default="test", help="Prefix for output files")
    parser.add_argument("-d", "--datadir", type=str, default=os.getcwd(), help="Output directory")
    parser.add_argument("-n", "--nvirion", type=int, default=1, help="Number of virions/clusters (for KMeans)")
    return parser.parse_args()


def load_xyz_file(filepath):
    """
    Safely load .xyz file.
    Supports formats with headers or atom labels (e.g. from convert_to_coords.py)
    """
    coords = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0].isalpha():  # skip headers like "C ..." or text lines
                parts = line.split()
                if len(parts) == 4 and parts[0].isalpha():  # has atom name + coords
                    try:
                        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except:
                        continue
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                try:
                    coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except:
                    pass
    return np.array(coords)


def inlier_outlier(cloud, ind, datadir, prefix):
    """Visualize inliers and outliers."""
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("[INFO] Showing outliers (red) and inliers (gray)...")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # headless mode (safe in Linux/servers)
    vis.add_geometry(inlier_cloud)
    vis.add_geometry(outlier_cloud)
    vis.poll_events()
    vis.update_renderer()
    out_path = os.path.join(datadir, f"{prefix}_inlier_outlier.png")
    vis.capture_screen_image(out_path)
    vis.destroy_window()
    print(f"[INFO] Saved visualization → {out_path}")


def main():
    args = parse_args()
    datadir, coords, prefix, nvirion = args.datadir, args.coordinates, args.prefix, args.nvirion

    # --- Ensure output directory exists ---
    os.makedirs(datadir, exist_ok=True)

    # --- Load Point Cloud ---
    print(f"[INFO] Reading {coords}")
    pcd = o3d.io.read_point_cloud(coords, format="xyz")

    # Fallback if Open3D failed (e.g., 0 points)
    if len(pcd.points) == 0:
        print("[WARN] Open3D failed to load .xyz properly. Using manual loader...")
        arr = load_xyz_file(coords)
        if len(arr) == 0:
            print("❌ No valid coordinates found in file.")
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr)

    pcd.paint_uniform_color([0.8, 0.8, 0.8])

    # --- Remove Outliers ---
    print("[INFO] Cleaning point cloud (removing outliers)...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.1)
    inlier_outlier(pcd, ind, datadir, prefix)

    # --- Save Cleaned Points ---
    cl_array = np.asarray(cl.points)
    df = pd.DataFrame(cl_array, columns=["x", "y", "z"])
    clean_path = os.path.join(datadir, f"{prefix}_glyco.csv")
    df.to_csv(clean_path, index=False)
    print(f"[INFO] Cleaned coordinates saved → {clean_path}")

    # --- Cluster into Virions (KMeans) ---
    if nvirion > 1:
        print(f"[INFO] Clustering into {nvirion} virions using KMeans...")
        y_pred = KMeans(n_clusters=nvirion, random_state=42).fit(cl_array)
        labels = y_pred.labels_

        pd.DataFrame(labels, columns=["label"]).to_csv(
            os.path.join(datadir, f"{prefix}_labels.csv"), index=False
        )

        for i in range(nvirion):
            idx = np.where(labels == i)[0]
            single_cloud = cl.select_by_index(idx)
            arr = np.asarray(single_cloud.points)

            subfile = os.path.join(datadir, f"{prefix}_{i}_glyco.csv")
            pd.DataFrame(arr, columns=["x", "y", "z"]).to_csv(subfile, index=False)

            # Estimate normals (required for Poisson reconstruction)
            single_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=100)
            )
            single_cloud.orient_normals_consistent_tangent_plane(100)

            print(f"[INFO] Running Poisson surface reconstruction for cluster {i}...")
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(single_cloud, depth=9)
            mesh.compute_vertex_normals()

            mesh_path = os.path.join(datadir, f"{prefix}_{i}_mesh.stl")
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f" → saved mesh {mesh_path}")


# Usage:
# python3 data/influenza_morphological_glyco.py data/xyz/emd_46043_NA.xyz --prefix NA_clean --datadir ./data/NA_data_items --nvirion 3
# python3 data/influenza_morphological_glyco.py data/xyz/emd_0025_HA.xyz --prefix HA_clean --datadir ./data/HA_data_items --nvirion 3
if __name__ == "__main__":
    main()
