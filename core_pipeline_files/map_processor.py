#!/usr/bin/env python3
"""
map_processor.py
---------------
Process cryo-EM maps and PDB structures through the complete pipeline.
Supports .map, .mrc, .pdb, .ent, and .cif files.

KEY UPDATE: PDB files are now converted to realistic cryo-EM maps before
feature extraction, ensuring biologically meaningful results.
"""

import os
import pandas as pd
from typing import Dict, List
from processing_modules import (
    UnifiedConverter, PointCloudCleaner, VirionClusterer,
    NeighborAnalyzer, MeshGenerator, FeatureExtractor
)
from pipeline_config import (
    PROCESSING_PARAMS, get_protein_type, get_class_label, 
    TRAINING_FEATURE_COLS, CACHE_DIR
)


class MapProcessor:
    """Process map/PDB files through the entire pipeline."""
    
    def __init__(self, params: Dict = None, cache_dir: str = None):
        self.params = params or PROCESSING_PARAMS
        self.cache_dir = cache_dir or CACHE_DIR
        
        # Initialize processing modules
        # Pass cache_dir to UnifiedConverter for PDB map caching
        self.converter = UnifiedConverter(cache_dir=self.cache_dir)
        self.cleaner = PointCloudCleaner()
        self.clusterer = VirionClusterer()
        self.neighbor_analyzer = NeighborAnalyzer()
        self.mesh_generator = MeshGenerator()
        self.feature_extractor = FeatureExtractor()
    
    def _get_file_type(self, filepath: str) -> str:
        """Determine file type from extension."""
        ext = filepath.lower().split('.')[-1]
        if ext in ['map', 'mrc']:
            return 'density_map'
        elif ext in ['pdb', 'ent', 'cif']:
            return 'atomic_structure'
        else:
            return 'unknown'
    
    def process(self, map_file: str, verbose: bool = True) -> List[Dict]:
        """
        Process a single file and extract features.
        
        For PDB files: Automatically converts to realistic cryo-EM map first
        For MAP files: Processes directly
        
        Returns:
            List[Dict]: List of feature dictionaries (one per cluster)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing: {os.path.basename(map_file)}")
            print(f"{'='*70}")
        
        # Extract metadata
        filename = os.path.basename(map_file)
        file_type = self._get_file_type(map_file)
        protein_type = get_protein_type(filename)
        class_label = get_class_label(protein_type)
        
        if verbose:
            print(f"[1/6] File type: {file_type}")
            print(f"      Detected protein type: {protein_type} (class {class_label})")
        
        # Step 1: Convert to coordinates
        if verbose:
            print(f"[2/6] Converting to coordinates...")
        
        try:
            if file_type == 'density_map':
                # Direct conversion for density maps
                coords = self.converter.process(
                    map_file,
                    threshold=self.params["threshold"],
                    downsample=self.params["downsample"]
                )
            elif file_type == 'atomic_structure':
                # PDB → Realistic Map → Coordinates
                # This ensures biologically meaningful features
                if verbose:
                    print(f"      [PDB] Will convert to realistic cryo-EM map first")
                
                coords = self.converter.process(
                    map_file,
                    threshold=self.params["threshold"],
                    downsample=self.params["downsample"],
                    # Additional PDB-specific parameters
                    apix=self.params.get("apix", 1.2),
                    resolution=self.params.get("resolution", 6.0),
                    box_size=self.params.get("box_size", 256),
                    n_virions=self.params["n_clusters"],  # Match clustering
                    min_separation=self.params.get("min_separation", 150.0),
                    noise_level=self.params.get("noise_level", 2.0)
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            print(f"[ERROR] Failed to convert file: {e}")
            raise
        
        if verbose:
            print(f"      Extracted {len(coords)} points")
        
        # Step 2: Clean point cloud
        if verbose:
            print(f"[3/6] Cleaning point cloud...")
        cleaned_coords = self.cleaner.clean(
            coords,
            nb_neighbors=self.params["nb_neighbors"],
            std_ratio=self.params["std_ratio"]
        )
        if verbose:
            print(f"      Retained {len(cleaned_coords)} points after cleaning")
        
        # Step 3: Cluster into virions
        if verbose:
            print(f"[4/6] Clustering into {self.params['n_clusters']} virions...")
        clusters, labels = self.clusterer.cluster(
            cleaned_coords,
            n_clusters=self.params["n_clusters"]
        )
        if verbose:
            print(f"      Created {len(clusters)} clusters")
        
        # Step 4-6: Process each cluster
        features_list = []
        for i, cluster_coords in enumerate(clusters):
            if verbose:
                print(f"[5/6] Processing cluster {i+1}/{len(clusters)}...")
            
            # Compute neighbor statistics
            neighbor_stats = self.neighbor_analyzer.compute_distances(
                cluster_coords,
                k=self.params["k_neighbors"],
                kth_ignore=self.params["kth_ignore"],
                pixel_size=self.params["pixel_size"]
            )
            
            # Generate mesh
            mesh = self.mesh_generator.generate(
                cluster_coords,
                depth=self.params["poisson_depth"],
                normal_radius=self.params["normal_radius"],
                normal_max_nn=self.params["normal_max_nn"]
            )
            
            # Extract features (ONLY training features)
            features = self.feature_extractor.extract(
                cluster_coords,
                mesh,
                neighbor_stats
            )
            
            # Add class label (the only metadata we keep)
            features["class_label"] = class_label
            
            features_list.append(features)
            
            if verbose:
                print(f"      Cluster {i+1}: {features['num_points']} points, "
                      f"density {features['density']:.4f}, "
                      f"bbox_volume {features['bbox_volume']:.2f}")
        
        if verbose:
            print(f"[6/6] ✓ Completed processing {filename}")
            print(f"      Generated {len(features_list)} feature sets")
        
        return features_list
    
    def process_batch(self, map_files: List[str], verbose: bool = True) -> pd.DataFrame:
        """
        Process multiple files.
        
        Returns:
            pd.DataFrame: Combined features from all files (column-ordered)
        """
        all_features = []
        
        for i, map_file in enumerate(map_files, 1):
            if verbose:
                print(f"\n{'#'*70}")
                print(f"File {i}/{len(map_files)}")
                print(f"{'#'*70}")
            
            try:
                features = self.process(map_file, verbose=verbose)
                all_features.extend(features)
            except Exception as e:
                print(f"[ERROR] Failed to process {map_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Convert to DataFrame with ONLY training features
        df = pd.DataFrame(all_features)
        
        # Ensure correct column order (as specified in TRAINING_FEATURE_COLS)
        df = df[TRAINING_FEATURE_COLS]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"BATCH PROCESSING COMPLETE")
            print(f"{'='*70}")
            print(f"Total files processed: {len(map_files)}")
            print(f"Total feature rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")
        
        return df