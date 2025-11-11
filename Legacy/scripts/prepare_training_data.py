#!/usr/bin/env python3
"""
prepare_training_data.py
------------------------

Combine glyco-feature CSVs, nearest-neighbour distances, and cluster labels
into unified training datasets for HA and NA proteins.

Usage:
    python3 scripts/prepare_training_data.py
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "training_data")
os.makedirs(OUT_DIR, exist_ok=True)


def merge_features_and_labels(prefix: str):
    """Merge *_glyco.csv, *_nbr.csv, and *_labels.csv into one dataset."""
    print(f"\n[INFO] Processing {prefix} dataset...")

    # Define paths
    folder = os.path.join(DATA_DIR, f"{prefix}_data_items")
    glyco_path = os.path.join(folder, f"{prefix}_clean_glyco.csv")
    nbr_path = os.path.join(folder, f"{prefix}_nbr.csv")
    labels_path = os.path.join(folder, f"{prefix}_clean_labels.csv")

    # Load data
    glyco = pd.read_csv(glyco_path)
    nbr = pd.read_csv(nbr_path)
    labels = pd.read_csv(labels_path)

    # Ensure labels are in a single column (ignoring header row)
    if "label" in labels.columns:
        labels = labels["label"]
    else:
        labels = labels.iloc[:, 0]

    # Match row counts
    min_len = min(len(glyco), len(nbr), len(labels))
    glyco = glyco.iloc[:min_len, :]
    nbr = nbr.iloc[:min_len, :]
    labels = labels.iloc[:min_len]

    # Merge into one DataFrame
    df = pd.concat([glyco, nbr, labels.rename("label")], axis=1)

    # Save output
    out_path = os.path.join(OUT_DIR, f"{prefix}_training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved → {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df


def main():
    ha_df = merge_features_and_labels("HA")
    na_df = merge_features_and_labels("NA")

    # Combine both into one dataset
    combined = pd.concat([ha_df.assign(type="HA"), na_df.assign(type="NA")])
    combined_out = os.path.join(OUT_DIR, "combined_training_data.csv")
    combined.to_csv(combined_out, index=False)

    print(f"\n✅ Combined dataset saved → {combined_out}")
    print(f"   Total samples: {combined.shape[0]}")
    print(f"   Total features per sample: {combined.shape[1] - 2} (excluding label, type)")


if __name__ == "__main__":
    main()
