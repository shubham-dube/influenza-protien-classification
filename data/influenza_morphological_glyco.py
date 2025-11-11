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


def inlier_outlier(cloud, ind, datadir, prefix):
    """Visualize inliers and outliers."""
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray)...")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(inlier_cloud)
    vis.add_geometry(outlier_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(datadir, f"{prefix}_inlier_outlier.png"))
    vis.destroy_window()


def main():
    args = parse_args()
    datadir, coords, prefix, nvirion = args.datadir, args.coordinates, args.prefix, args.nvirion

    # --- Load Point Cloud ---
    print(f"[INFO] Reading {coords}")
    pcd = o3d.io.read_point_cloud(coords, format="xyz")
    pcd.paint_uniform_color([0.8, 0.8, 0.8])

    # --- Remove Outliers ---
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.1)
    inlier_outlier(pcd, ind, datadir, prefix)

    # --- Save Cleaned Points ---
    cl_array = np.asarray(cl.points)
    df = pd.DataFrame(cl_array, columns=["x", "y", "z"])
    df.to_csv(os.path.join(datadir, f"{prefix}_glyco.csv"), index=False)
    print(f"[INFO] Cleaned coordinates saved → {prefix}_glyco.csv")

    # --- Cluster into Virions (KMeans) ---
    if nvirion > 1:
        print(f"[INFO] Clustering into {nvirion} virions...")
        y_pred = KMeans(n_clusters=nvirion, random_state=42).fit(cl_array)
        labels = y_pred.labels_
        pd.DataFrame(labels, columns=["label"]).to_csv(os.path.join(datadir, f"{prefix}_labels.csv"), index=False)

        for i in range(nvirion):
            idx = np.where(labels == i)[0]
            single_cloud = cl.select_by_index(idx)
            arr = np.asarray(single_cloud.points)

            pd.DataFrame(arr, columns=["x", "y", "z"]).to_csv(
                os.path.join(datadir, f"{prefix}_{i}_glyco.csv"), index=False
            )

            single_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=100)
            )
            single_cloud.orient_normals_consistent_tangent_plane(100)

            print(f"[INFO] Poisson surface reconstruction for cluster {i}")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(single_cloud, depth=9)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(os.path.join(datadir, f"{prefix}_{i}_mesh.stl"), mesh)
            print(f" → saved mesh {prefix}_{i}_mesh.stl")

# python3 influenza_morphological_glyco.py path/to/input.xyz --prefix HA_clean --datadir ./output --nvirion 3
if __name__ == "__main__":
    main()
