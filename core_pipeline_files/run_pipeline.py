#!/usr/bin/env python3
"""
run_pipeline.py
--------------
Automated pipeline for processing cryo-EM maps and PDB structures.

Usage:
    python run_pipeline.py                    # Process all files in ./maps/ and ./pdb/
    python run_pipeline.py --clear-cache      # Clear cache and reprocess all
    python run_pipeline.py --skip-cache       # Process all without using cache
    python run_pipeline.py --maps-only        # Process only density maps
    python run_pipeline.py --pdb-only         # Process only PDB files
"""

import os
import sys
import glob
import argparse
import pandas as pd
from datetime import datetime
from typing import List

from core_pipeline_files.pipeline_config import (
    MAPS_DIR, PDB_DIR, OUTPUT_DIR, FINAL_DATASET, CACHE_INDEX_FILE,
    ENABLE_CACHING, ensure_directories, TRAINING_FEATURE_COLS,
    is_supported_file
)
from core_pipeline_files.cache_manager import CacheManager
from core_pipeline_files.map_processor import MapProcessor


def find_files(directory: str, extensions: List[str]) -> List[str]:
    """Find all files with given extensions in directory."""
    all_files = []
    
    for ext in extensions:
        pattern = f"*.{ext}"
        all_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    return sorted(all_files)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated cryo-EM map and PDB processing pipeline"
    )
    parser.add_argument(
        "--maps-dir", 
        type=str, 
        default=MAPS_DIR,
        help=f"Directory containing map files (default: {MAPS_DIR})"
    )
    parser.add_argument(
        "--pdb-dir", 
        type=str, 
        default=PDB_DIR,
        help=f"Directory containing PDB files (default: {PDB_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache and reprocess all files"
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip cache checking (process all files)"
    )
    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="Process only density map files (.map, .mrc)"
    )
    parser.add_argument(
        "--pdb-only",
        action="store_true",
        help="Process only PDB structure files (.pdb, .ent, .cif)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed processing information"
    )
    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_arguments()
    
    # Setup directories
    ensure_directories()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("AUTOMATED CRYO-EM/PDB PROCESSING PIPELINE (v3.0)")
    print("TRAINING FEATURES ONLY")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Maps directory: {args.maps_dir}")
    print(f"PDB directory: {args.pdb_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    # Initialize cache manager
    cache = CacheManager(CACHE_INDEX_FILE)
    
    if args.clear_cache:
        print("\n[INFO] Clearing cache...")
        cache.clear_cache()
    
    # Collect all input files
    all_files = []
    
    # Find density map files
    if not args.pdb_only:
        map_files = find_files(args.maps_dir, ['map', 'mrc'])
        all_files.extend(map_files)
        if map_files:
            print(f"\n[INFO] Found {len(map_files)} density map files (.map, .mrc)")
    
    # Find PDB structure files
    if not args.maps_only:
        pdb_files = find_files(args.pdb_dir, ['pdb', 'ent', 'cif'])
        all_files.extend(pdb_files)
        if pdb_files:
            print(f"[INFO] Found {len(pdb_files)} PDB structure files (.pdb, .ent, .cif)")
    
    if not all_files:
        print(f"\n[ERROR] No supported files found!")
        print(f"[INFO] Supported formats: .map, .mrc, .pdb, .ent, .cif")
        print(f"[INFO] Place files in:")
        print(f"       - Maps: {args.maps_dir}")
        print(f"       - PDBs: {args.pdb_dir}")
        sys.exit(1)
    
    print(f"\n[INFO] Total files found: {len(all_files)}")
    
    # Filter out already processed files (unless skip-cache is set)
    if ENABLE_CACHING and not args.skip_cache:
        unprocessed_files = []
        skipped_files = []
        
        for file_path in all_files:
            if cache.is_processed(file_path):
                skipped_files.append(file_path)
            else:
                unprocessed_files.append(file_path)
        
        if skipped_files:
            print(f"\n[INFO] Skipping {len(skipped_files)} already processed files:")
            for f in skipped_files[:5]:  # Show first 5
                print(f"  ✓ {os.path.basename(f)}")
            if len(skipped_files) > 5:
                print(f"  ... and {len(skipped_files) - 5} more")
        
        all_files = unprocessed_files
    
    if not all_files:
        print("\n[INFO] All files have been processed!")
        print("[INFO] Use --clear-cache to reprocess all files")
        
        # Load existing dataset
        if os.path.exists(FINAL_DATASET):
            print(f"\n[INFO] Loading existing dataset: {FINAL_DATASET}")
            df = pd.read_csv(FINAL_DATASET)
            print(f"[INFO] Dataset contains {len(df)} rows")
            print(f"[INFO] Columns: {list(df.columns)}")
            
            # Print class distribution
            print(f"\n[INFO] Class distribution:")
            for cls, count in df['class_label'].value_counts().items():
                print(f"   Class {cls}: {count} samples")
        
        sys.exit(0)
    
    print(f"\n[INFO] Processing {len(all_files)} new files...")
    
    # Group files by type for reporting
    map_count = sum(1 for f in all_files if f.lower().endswith(('.map', '.mrc')))
    pdb_count = sum(1 for f in all_files if f.lower().endswith(('.pdb', '.ent', '.cif')))
    print(f"       - Density maps: {map_count}")
    print(f"       - PDB structures: {pdb_count}")
    
    # Initialize processor
    processor = MapProcessor()
    
    # Process files
    new_features_df = processor.process_batch(all_files, verbose=args.verbose)
    
    # Mark files as processed in cache
    if ENABLE_CACHING:
        for file_path in all_files:
            num_samples = len(new_features_df)
            cache.mark_processed(file_path, {
                "num_samples": num_samples
            })
    
    # Load existing dataset if it exists
    if os.path.exists(FINAL_DATASET):
        print(f"\n[INFO] Loading existing dataset: {FINAL_DATASET}")
        existing_df = pd.read_csv(FINAL_DATASET)
        print(f"[INFO] Existing dataset: {len(existing_df)} rows")
        
        # Combine with new data
        combined_df = pd.concat([existing_df, new_features_df], ignore_index=True)
        print(f"[INFO] Combined dataset: {len(combined_df)} rows")
    else:
        combined_df = new_features_df
    
    # Save final dataset
    combined_df.to_csv(FINAL_DATASET, index=False)
    print(f"\n✅ Saved final dataset: {FINAL_DATASET}")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Total columns: {len(combined_df.columns)}")
    
    # Save detailed summary
    summary_file = os.path.join(args.output_dir, "dataset_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DATASET SUMMARY (v3.0 - TRAINING FEATURES ONLY)\n")
        f.write("Supports: .map, .mrc, .pdb, .ent, .cif files\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {len(combined_df)}\n\n")
        
        f.write("Samples by class label:\n")
        class_counts = combined_df["class_label"].value_counts()
        for class_label, count in class_counts.items():
            f.write(f"  Class {class_label}: {count}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("TRAINING FEATURES (in order):\n")
        f.write("="*70 + "\n")
        for col in TRAINING_FEATURE_COLS:
            if col in combined_df.columns:
                f.write(f"  - {col}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("FEATURE STATISTICS:\n")
        f.write("="*70 + "\n")
        
        # Get training features only (excluding class_label)
        feature_cols = [c for c in TRAINING_FEATURE_COLS if c != 'class_label']
        
        for col in feature_cols:
            f.write(f"\n{col}:\n")
            f.write(f"  Mean: {combined_df[col].mean():.6f}\n")
            f.write(f"  Std:  {combined_df[col].std():.6f}\n")
            f.write(f"  Min:  {combined_df[col].min():.6f}\n")
            f.write(f"  Max:  {combined_df[col].max():.6f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✅ Saved summary: {summary_file}")
    
    # Print cache statistics
    if ENABLE_CACHING:
        stats = cache.get_statistics()
        print(f"\n[INFO] Cache statistics:")
        print(f"   Total processed files: {stats['total_processed']}")
    
    # Print feature info
    print(f"\n[INFO] Final Feature Set ({len([c for c in TRAINING_FEATURE_COLS if c != 'class_label'])} features):")
    for feat in [c for c in TRAINING_FEATURE_COLS if c != 'class_label']:
        print(f"   ✓ {feat}")
    print(f"   ✓ class_label (target)")
    
    print(f"\n[INFO] Feature Engineering Applied:")
    print(f"   • density = num_points / surface_area")
    print(f"   • bbox_volume = coord_range_x × coord_range_y × coord_range_z")
    
    print(f"\n[INFO] Supported Input Formats:")
    print(f"   • Density maps: .map, .mrc")
    print(f"   • Atomic structures: .pdb, .ent, .cif")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nReady for model training with: {FINAL_DATASET}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)