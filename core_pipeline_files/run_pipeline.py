#!/usr/bin/env python3
"""
run_pipeline.py
--------------
Automated pipeline for processing multiple cryo-EM maps.

Usage:
    python run_pipeline.py                    # Process all maps in ./maps/
    python run_pipeline.py --clear-cache      # Clear cache and reprocess all
    python run_pipeline.py --skip-cache       # Process all without using cache
"""

import os
import sys
import glob
import argparse
import pandas as pd
from datetime import datetime
from typing import List

from pipeline_config import (
    MAPS_DIR, OUTPUT_DIR, FINAL_DATASET, CACHE_INDEX_FILE,
    ENABLE_CACHING, ensure_directories
)
from cache_manager import CacheManager
from map_processor import MapProcessor


def find_map_files(directory: str) -> List[str]:
    """Find all .map and .mrc files in directory."""
    patterns = ["*.map", "*.mrc"]
    map_files = []
    
    for pattern in patterns:
        map_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    return sorted(map_files)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated cryo-EM map processing pipeline"
    )
    parser.add_argument(
        "--maps-dir", 
        type=str, 
        default=MAPS_DIR,
        help=f"Directory containing map files (default: {MAPS_DIR})"
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
        help="Clear cache and reprocess all maps"
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip cache checking (process all maps)"
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
    print("AUTOMATED CRYO-EM PROCESSING PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Maps directory: {args.maps_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    # Initialize cache manager
    cache = CacheManager(CACHE_INDEX_FILE)
    
    if args.clear_cache:
        print("\n[INFO] Clearing cache...")
        cache.clear_cache()
    
    # Find all map files
    map_files = find_map_files(args.maps_dir)
    
    if not map_files:
        print(f"\n[ERROR] No .map or .mrc files found in {args.maps_dir}")
        print("[INFO] Please place your map files in the 'maps' directory")
        sys.exit(1)
    
    print(f"\n[INFO] Found {len(map_files)} map files")
    
    # Filter out already processed maps (unless skip-cache is set)
    if ENABLE_CACHING and not args.skip_cache:
        unprocessed_maps = []
        skipped_maps = []
        
        for map_file in map_files:
            if cache.is_processed(map_file):
                skipped_maps.append(map_file)
            else:
                unprocessed_maps.append(map_file)
        
        if skipped_maps:
            print(f"\n[INFO] Skipping {len(skipped_maps)} already processed maps:")
            for f in skipped_maps:
                print(f"  ✓ {os.path.basename(f)}")
        
        map_files = unprocessed_maps
    
    if not map_files:
        print("\n[INFO] All maps have been processed!")
        print("[INFO] Use --clear-cache to reprocess all maps")
        
        # Load existing dataset
        if os.path.exists(FINAL_DATASET):
            print(f"\n[INFO] Loading existing dataset: {FINAL_DATASET}")
            df = pd.read_csv(FINAL_DATASET)
            print(f"[INFO] Dataset contains {len(df)} rows")
            print(f"[INFO] Columns: {list(df.columns)}")
        
        sys.exit(0)
    
    print(f"\n[INFO] Processing {len(map_files)} new maps...")
    
    # Initialize processor
    processor = MapProcessor()
    
    # Process maps
    new_features_df = processor.process_batch(map_files, verbose=args.verbose)
    
    # Mark maps as processed in cache
    if ENABLE_CACHING:
        for map_file in map_files:
            cache.mark_processed(map_file, {
                "num_clusters": len(new_features_df[
                    new_features_df["map_file"] == os.path.basename(map_file)
                ])
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
    
    # Save summary by protein type
    summary_file = os.path.join(args.output_dir, "dataset_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DATASET SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {len(combined_df)}\n\n")
        
        f.write("Samples by protein type:\n")
        type_counts = combined_df["protein_type"].value_counts()
        for protein_type, count in type_counts.items():
            f.write(f"  {protein_type}: {count}\n")
        
        f.write("\nSamples by class label:\n")
        class_counts = combined_df["class_label"].value_counts()
        for class_label, count in class_counts.items():
            f.write(f"  Class {class_label}: {count}\n")
        
        f.write("\nFeature columns:\n")
        for col in combined_df.columns:
            f.write(f"  - {col}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✅ Saved summary: {summary_file}")
    
    # Print cache statistics
    if ENABLE_CACHING:
        stats = cache.get_statistics()
        print(f"\n[INFO] Cache statistics:")
        print(f"   Total processed maps: {stats['total_processed']}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


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