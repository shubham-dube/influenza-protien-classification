#!/usr/bin/env python3
"""
utils.py
-------
Utility functions for dataset analysis and visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def analyze_dataset(csv_path: str) -> Dict:
    """
    Analyze the final dataset and return statistics.
    
    Args:
        csv_path: Path to the dataset CSV
        
    Returns:
        Dictionary with analysis results
    """
    df = pd.read_csv(csv_path)
    
    analysis = {
        "total_samples": len(df),
        "total_maps": df["map_file"].nunique(),
        "protein_types": df["protein_type"].value_counts().to_dict(),
        "class_distribution": df["class_label"].value_counts().to_dict(),
        "clusters_per_map": df.groupby("map_file").size().describe().to_dict(),
        "feature_statistics": {},
    }
    
    # Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "class_label" and col != "cluster_id":
            analysis["feature_statistics"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
    
    return analysis


def export_for_ml(csv_path: str, output_path: str, 
                  exclude_cols: List[str] = None) -> None:
    """
    Export dataset in ML-ready format (features + labels).
    
    Args:
        csv_path: Input dataset path
        output_path: Output path for ML dataset
        exclude_cols: Columns to exclude from features
    """
    df = pd.read_csv(csv_path)
    
    # Default columns to exclude
    if exclude_cols is None:
        exclude_cols = ["map_file", "protein_type", "cluster_id"]
    
    # Separate features and labels
    feature_cols = [c for c in df.columns 
                   if c not in exclude_cols and c != "class_label"]
    
    X = df[feature_cols]
    y = df["class_label"]
    
    # Combine
    ml_df = X.copy()
    ml_df["label"] = y
    
    ml_df.to_csv(output_path, index=False)
    print(f"✅ Exported ML-ready dataset: {output_path}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(ml_df)}")


def split_train_test(csv_path: str, output_dir: str, 
                     test_size: float = 0.2, random_state: int = 42) -> None:
    """
    Split dataset into train/test sets.
    
    Args:
        csv_path: Input dataset path
        output_dir: Directory for train/test files
        test_size: Fraction for test set
        random_state: Random seed
    """
    import os
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(csv_path)
    
    # Split by map file (to avoid data leakage)
    unique_maps = df["map_file"].unique()
    train_maps, test_maps = train_test_split(
        unique_maps, test_size=test_size, random_state=random_state
    )
    
    train_df = df[df["map_file"].isin(train_maps)]
    test_df = df[df["map_file"].isin(test_maps)]
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train_dataset.csv")
    test_path = os.path.join(output_dir, "test_dataset.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Train set: {train_path} ({len(train_df)} samples)")
    print(f"✅ Test set: {test_path} ({len(test_df)} samples)")
    print(f"   Train maps: {len(train_maps)}")
    print(f"   Test maps: {len(test_maps)}")


def print_dataset_summary(csv_path: str) -> None:
    """Print a formatted summary of the dataset."""
    analysis = analyze_dataset(csv_path)
    
    print("="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Total maps: {analysis['total_maps']}")
    
    print("\nProtein type distribution:")
    for ptype, count in analysis['protein_types'].items():
        print(f"  {ptype}: {count} samples")
    
    print("\nClass label distribution:")
    for label, count in analysis['class_distribution'].items():
        print(f"  Class {label}: {count} samples")
    
    print("\nClusters per map:")
    for stat, val in analysis['clusters_per_map'].items():
        print(f"  {stat}: {val:.2f}")
    
    print("\nTop features (by std deviation):")
    feature_stats = analysis['feature_statistics']
    sorted_features = sorted(
        feature_stats.items(), 
        key=lambda x: x[1]['std'], 
        reverse=True
    )[:5]
    
    for feat, stats in sorted_features:
        print(f"  {feat}:")
        print(f"    mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    print("="*70)


def check_data_quality(csv_path: str) -> Dict:
    """
    Check dataset for potential quality issues.
    
    Returns:
        Dictionary with quality metrics
    """
    df = pd.read_csv(csv_path)
    
    issues = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": len(df) - len(df.drop_duplicates()),
        "zero_values": {},
        "outliers": {},
    }
    
    # Check for zero values in important features
    for col in ["num_points", "aspect_ratio", "volume"]:
        if col in df.columns:
            issues["zero_values"][col] = (df[col] == 0).sum()
    
    # Simple outlier detection (3 sigma rule)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["class_label", "cluster_id"]:
            mean = df[col].mean()
            std = df[col].std()
            outliers = ((df[col] < mean - 3*std) | (df[col] > mean + 3*std)).sum()
            if outliers > 0:
                issues["outliers"][col] = outliers
    
    return issues


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python utils.py <dataset.csv> [command]")
        print("\nCommands:")
        print("  summary      - Print dataset summary (default)")
        print("  quality      - Check data quality")
        print("  export       - Export ML-ready format")
        print("  split        - Split into train/test sets")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else "summary"
    
    if command == "summary":
        print_dataset_summary(csv_path)
    
    elif command == "quality":
        issues = check_data_quality(csv_path)
        print("Data Quality Check:")
        print(f"  Duplicate rows: {issues['duplicate_rows']}")
        if issues['outliers']:
            print(f"  Outliers detected in: {list(issues['outliers'].keys())}")
    
    elif command == "export":
        output = csv_path.replace(".csv", "_ml_ready.csv")
        export_for_ml(csv_path, output)
    
    elif command == "split":
        split_train_test(csv_path, "output/splits")
    
    else:
        print(f"Unknown command: {command}")