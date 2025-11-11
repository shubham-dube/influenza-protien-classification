# #!/usr/bin/env python3
# """
# Phase 2: Data Labeling and Feature Engineering for Influenza Proteins
# Generates training_data.csv for ML models.
# """

# import os
# import argparse
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# import open3d as o3d

# LABELS = {"HA": 0, "NA": 1}

# def parse_args():
#     parser = argparse.ArgumentParser(description="Feature engineering for influenza morphology datasets.")
#     parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing *_glyco.csv files or subfolders")
#     parser.add_argument("-o", "--output", type=str, default="training_data.csv", help="Output training CSV")
#     return parser.parse_args()

# def compute_features(csv_path):
#     """Compute geometric and spatial features from point cloud."""
#     data = pd.read_csv(csv_path)
#     if not {"x", "y", "z"}.issubset(data.columns):
#         print(f"[WARN] Missing x,y,z columns in {csv_path}")
#         return None

#     points = data[["x", "y", "z"]].values
#     n_points = len(points)
#     if n_points < 2:
#         return None

#     # --- Nearest Neighbor distances ---
#     nbrs = NearestNeighbors(n_neighbors=2).fit(points)
#     distances, _ = nbrs.kneighbors(points)
#     nn_dists = distances[:, 1]  # skip self
#     mean_dist = np.mean(nn_dists)
#     std_dist = np.std(nn_dists)

#     # --- Aspect Ratio ---
#     min_vals = np.min(points, axis=0)
#     max_vals = np.max(points, axis=0)
#     lengths = max_vals - min_vals
#     aspect_ratio = np.max(lengths) / np.min(lengths) if np.min(lengths) > 0 else 0

#     # --- Density (approximation) ---
#     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
#     try:
#         hull, _ = pcd.compute_convex_hull()
#         area = hull.get_surface_area()
#         density = n_points / area if area > 0 else 0
#     except Exception:
#         density = 0

#     return [mean_dist, std_dist, aspect_ratio, n_points, density]

# def main():
#     args = parse_args()
#     all_features = []

#     print(f"[INFO] Scanning directory: {args.input_dir}")

#     for root, dirs, files in os.walk(args.input_dir):  # ✅ searches subfolders too
#         for file in files:
#             if file.endswith("_glyco.csv"):
#                 prefix = os.path.basename(file).split("_")[0]  # e.g. HA, NA, M1
#                 label = LABELS.get(prefix.upper(), -1)
#                 if label == -1:
#                     print(f"[WARN] Unknown label for {file}")
#                     continue

#                 csv_path = os.path.join(root, file)
#                 print(f"[INFO] Processing {csv_path} → class {label}")
#                 features = compute_features(csv_path)
#                 if features:
#                     all_features.append(features + [label])

#     if not all_features:
#         print("[ERROR] No valid *_glyco.csv files found or all were invalid!")
#         return

#     df = pd.DataFrame(all_features, columns=[
#         "mean_dist", "std_dist", "aspect_ratio", "num_points", "density", "class_label"
#     ])

#     os.makedirs(os.path.dirname(args.output), exist_ok=True)
#     df.to_csv(args.output, index=False)

#     # Summary
#     print(f"\n[✅] Saved training dataset → {args.output}")
#     print(f"[INFO] Total samples: {len(df)}")
#     print(df.groupby("class_label").size())

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Feature Engineering (Per Point Version)
Each glycan point (x, y, z) from each *_glyco.csv file becomes one training row.
Adds nearest-neighbour distance and local density features.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

LABELS = {
    "HA": 0,
    "NA": 1,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Per-point feature engineering for influenza morphology datasets.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing *_glyco.csv files")
    parser.add_argument("-o", "--output", type=str, default="training_data_per_point.csv", help="Output CSV file")
    return parser.parse_args()

def compute_per_point_features(csv_path):
    data = pd.read_csv(csv_path)
    if not {"x", "y", "z"}.issubset(data.columns):
        print(f"[WARN] Skipping {csv_path} – missing coordinates")
        return None

    points = data[["x", "y", "z"]].values
    n_points = len(points)
    if n_points < 5:
        return None

    # --- Local geometry using Nearest Neighbors ---
    nbrs = NearestNeighbors(n_neighbors=min(6, n_points)).fit(points)
    distances, indices = nbrs.kneighbors(points)

    # 1st neighbor distance
    first_nn = distances[:, 1]  # exclude self (index 0)
    # mean distance to 5 nearest neighbors (local crowding)
    local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)

    # build DataFrame
    df_feat = pd.DataFrame({
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "nn_dist": first_nn,
        "local_density": local_density
    })
    return df_feat

def main():
    args = parse_args()
    all_data = []

    for file in os.listdir(args.input_dir):
        if file.endswith("_glyco.csv"):
            prefix = os.path.basename(file).split("_")[0]  # e.g., HA, NA
            label = LABELS.get(prefix.upper(), -1)
            if label == -1:
                print(f"[WARN] Unknown label for {file}")
                continue

            path = os.path.join(args.input_dir, file)
            print(f"[INFO] Processing {file}")
            df_feat = compute_per_point_features(path)
            if df_feat is not None:
                df_feat["class_label"] = label
                df_feat["source_file"] = file
                all_data.append(df_feat)

    if not all_data:
        print("[ERROR] No valid data processed.")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    df_all.to_csv(args.output, index=False)
    print(f"[✅] Saved per-point training data → {args.output}")
    print(f"Total samples: {len(df_all)}")

if __name__ == "__main__":
    main()