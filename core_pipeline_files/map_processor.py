#!/usr/bin/env python3
"""
map_processor.py
---------------
Process a single cryo-EM map through the complete pipeline.
"""

import os
import pandas as pd
from typing import Dict, List
from processing_modules import (
    MapConverter, PointCloudCleaner, VirionClusterer,
    NeighborAnalyzer, MeshGenerator, FeatureExtractor
)
from pipeline_config import PROCESSING_PARAMS, get_protein_type, get_class_label


class MapProcessor:
    """Process a single map file through the entire pipeline."""
    
    def __init__(self, params: Dict = None):
        self.params = params or PROCESSING_PARAMS
        
        # Initialize processing modules
        self.converter = MapConverter()
        self.cleaner = PointCloudCleaner()
        self.clusterer = VirionClusterer()
        self.neighbor_analyzer = NeighborAnalyzer()
        self.mesh_generator = MeshGenerator()
        self.feature_extractor = FeatureExtractor()
    
    def process(self, map_file: str, verbose: bool = True) -> List[Dict]:
        """
        Process a single map file and extract features.
        
        Returns:
            List[Dict]: List of feature dictionaries (one per cluster)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing: {os.path.basename(map_file)}")
            print(f"{'='*70}")
        
        # Extract protein type and class
        protein_type = get_protein_type(os.path.basename(map_file))
        class_label = get_class_label(protein_type)
        
        if verbose:
            print(f"[1/6] Detected protein type: {protein_type} (class {class_label})")
        
        # Step 1: Convert map to coordinates
        if verbose:
            print(f"[2/6] Converting map to coordinates...")
        coords = self.converter.process(
            map_file,
            threshold=self.params["threshold"],
            downsample=self.params["downsample"]
        )
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
            
            # Extract features
            features = self.feature_extractor.extract(
                cluster_coords,
                mesh,
                neighbor_stats
            )
            
            # Add metadata
            features["map_file"] = os.path.basename(map_file)
            features["protein_type"] = protein_type
            features["class_label"] = class_label
            features["cluster_id"] = i
            
            features_list.append(features)
            
            if verbose:
                print(f"      Cluster {i+1}: {features['num_points']} points, "
                      f"aspect ratio {features['aspect_ratio']:.2f}")
        
        if verbose:
            print(f"[6/6] ✓ Completed processing {os.path.basename(map_file)}")
            print(f"      Generated {len(features_list)} feature sets")
        
        return features_list
    
    def process_batch(self, map_files: List[str], verbose: bool = True) -> pd.DataFrame:
        """
        Process multiple map files.
        
        Returns:
            pd.DataFrame: Combined features from all maps
        """
        all_features = []
        
        for i, map_file in enumerate(map_files, 1):
            if verbose:
                print(f"\n{'#'*70}")
                print(f"Map {i}/{len(map_files)}")
                print(f"{'#'*70}")
            
            try:
                features = self.process(map_file, verbose=verbose)
                all_features.extend(features)
            except Exception as e:
                print(f"[ERROR] Failed to process {map_file}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"BATCH PROCESSING COMPLETE")
            print(f"{'='*70}")
            print(f"Total maps processed: {len(map_files)}")
            print(f"Total feature rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")
        
        return df