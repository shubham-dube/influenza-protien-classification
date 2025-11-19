#!/usr/bin/env python3
"""
processing_modules.py
--------------------
Modular processing functions for cryo-EM data pipeline.
Now supports both density maps (.map/.mrc) and atomic structures (.pdb/.ent)
WITH AUTOMATIC PDB→MAP CONVERSION FOR REALISTIC FEATURE EXTRACTION
"""

import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, List
import mrcfile
from Bio.PDB import PDBParser, MMCIFParser
import warnings
import os

# Import the PDB-to-map converter
from pdb_to_realistic_map import PDBToRealisticMap


# ============================================================================
# MODULE 1A: MAP TO COORDINATES (Density Maps)
# ============================================================================
class MapConverter:
    """Convert cryo-EM density maps to 3D coordinates."""
    
    @staticmethod
    def process(filepath: str, threshold: float = 0.1, 
                downsample: int = 4) -> np.ndarray:
        """
        Extract coordinates from .map/.mrc file.
        
        Returns:
            np.ndarray: Nx3 array of (x, y, z) coordinates
        """
        with mrcfile.open(filepath, permissive=True) as mrc:
            data = mrc.data.copy()
            
            # Get voxel size
            try:
                if hasattr(mrc, "voxel_size"):
                    vs = mrc.voxel_size
                    if hasattr(vs, "x"):
                        voxel_size = float(np.mean([vs.x, vs.y, vs.z]))
                    else:
                        voxel_size = float(np.mean(vs))
                else:
                    voxel_size = 1.0
            except Exception:
                voxel_size = 1.0
        
        # Extract coordinates above threshold
        coords = np.argwhere(data > threshold)
        coords = coords[::downsample]
        coords = coords * voxel_size
        
        return coords


# ============================================================================
# MODULE 1B: PDB TO MAP TO COORDINATES (Atomic Structures)
# ============================================================================
class PDBConverter:
    """
    Convert PDB/ENT atomic structures to realistic cryo-EM maps,
    then extract coordinates.
    
    CRITICAL: We do NOT extract features directly from PDB coordinates!
    Instead, we:
    1. Convert PDB → Realistic cryo-EM map (with noise, CTF, multiple virions)
    2. Extract coordinates from that map (same as density map processing)
    
    This ensures biologically meaningful features.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize PDB converter with map generation.
        
        Args:
            cache_dir: Directory to cache converted maps
        """
        self.map_converter_engine = PDBToRealisticMap(cache_dir=cache_dir)
        self.density_extractor = MapConverter()
    
    def process(self, filepath: str, threshold: float = 0.1,
                downsample: int = 4, **kwargs) -> np.ndarray:
        """
        Convert PDB to realistic map, then extract coordinates.
        
        Args:
            filepath: Path to PDB/ENT/CIF file
            threshold: Density threshold for coordinate extraction
            downsample: Downsampling factor
            **kwargs: Additional parameters for map generation
        
        Returns:
            np.ndarray: Nx3 array of (x, y, z) coordinates
        """
        print(f"[PDB] Processing {os.path.basename(filepath)}")
        print(f"[PDB] Converting to realistic cryo-EM map first...")
        
        # Step 1: Convert PDB → Realistic Map
        # This automatically creates 3 virions with proper spacing
        map_file = self.map_converter_engine.convert(
            filepath,
            output_file=None,  # Use cache
            **kwargs
        )
        
        print(f"[PDB] ✓ Map generated: {os.path.basename(map_file)}")
        print(f"[PDB] Extracting coordinates from map...")
        
        # Step 2: Extract coordinates from the generated map
        coords = self.density_extractor.process(
            map_file,
            threshold=threshold,
            downsample=downsample
        )
        
        print(f"[PDB] ✓ Extracted {len(coords)} coordinates")
        
        return coords


# ============================================================================
# MODULE 1C: UNIFIED CONVERTER (Dispatcher)
# ============================================================================
class UnifiedConverter:
    """
    Unified converter that handles both density maps and PDB files.
    
    KEY CHANGE: PDB files are now converted to realistic maps FIRST,
    then processed identically to regular density maps.
    """
    
    def __init__(self, cache_dir: str = None):
        self.map_converter = MapConverter()
        self.pdb_converter = PDBConverter(cache_dir=cache_dir)
    
    def process(self, filepath: str, **kwargs) -> np.ndarray:
        """
        Process file based on extension.
        
        Args:
            filepath: Path to input file
            **kwargs: Additional parameters passed to specific converter
        
        Returns:
            np.ndarray: Nx3 array of coordinates
        """
        ext = filepath.lower().split('.')[-1]
        
        if ext in ['map', 'mrc']:
            # Density map processing (direct)
            threshold = kwargs.get('threshold', 0.1)
            downsample = kwargs.get('downsample', 4)
            return self.map_converter.process(filepath, threshold, downsample)
        
        elif ext in ['pdb', 'ent', 'cif']:
            # PDB processing (via realistic map conversion)
            # Extract relevant parameters
            threshold = kwargs.get('threshold', 0.1)
            downsample = kwargs.get('downsample', 4)
            
            # Map generation parameters
            map_params = {
                'apix': kwargs.get('apix', 1.2),
                'resolution': kwargs.get('resolution', 6.0),
                'box_size': kwargs.get('box_size', 256),
                'n_virions': kwargs.get('n_virions', 3),
                'min_separation': kwargs.get('min_separation', 150.0),
                'noise_level': kwargs.get('noise_level', 2.0),
            }
            
            return self.pdb_converter.process(
                filepath, 
                threshold=threshold,
                downsample=downsample,
                **map_params
            )
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")


# ============================================================================
# MODULE 2: POINT CLOUD CLEANING
# ============================================================================
class PointCloudCleaner:
    """Clean and preprocess point cloud data."""
    
    @staticmethod
    def clean(coords: np.ndarray, nb_neighbors: int = 200, 
              std_ratio: float = 0.1) -> np.ndarray:
        """
        Remove outliers from point cloud.
        
        Returns:
            np.ndarray: Cleaned coordinates
        """
        if len(coords) < nb_neighbors:
            # Adjust nb_neighbors if we have fewer points
            nb_neighbors = max(min(len(coords) - 1, 20), 1)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        cl, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio
        )
        
        return np.asarray(cl.points)


# ============================================================================
# MODULE 3: CLUSTERING
# ============================================================================
class VirionClusterer:
    """Cluster points into separate virions."""
    
    @staticmethod
    def cluster(coords: np.ndarray, n_clusters: int = 3) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Cluster coordinates using KMeans.
        
        Returns:
            Tuple[List[np.ndarray], np.ndarray]: 
                - List of coordinate arrays per cluster
                - Array of cluster labels
        """
        if len(coords) < n_clusters:
            return [coords], np.zeros(len(coords), dtype=int)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coords)
        
        clusters = []
        for i in range(n_clusters):
            cluster_coords = coords[labels == i]
            if len(cluster_coords) > 0:
                clusters.append(cluster_coords)
        
        return clusters, labels


# ============================================================================
# MODULE 4: NEAREST NEIGHBORS
# ============================================================================
class NeighborAnalyzer:
    """Compute nearest neighbor statistics."""
    
    @staticmethod
    def compute_distances(coords: np.ndarray, k: int = 3, 
                         kth_ignore: int = 1, 
                         pixel_size: float = 0.209) -> Dict[str, float]:
        """
        Compute nearest neighbor distance statistics.
        
        Returns:
            Dict with keys: mean_dist, std_dist, min_dist, max_dist
        """
        n_points = len(coords)
        if n_points < k + kth_ignore:
            return {
                "mean_dist": 0.0,
                "std_dist": 0.0,
                "min_dist": 0.0,
                "max_dist": 0.0
            }
        
        K_corr = min(k + kth_ignore, n_points)
        nbrs = NearestNeighbors(n_neighbors=K_corr, algorithm="auto").fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        
        # Remove self-distances
        distances = np.delete(distances, np.s_[:kth_ignore:], axis=1)
        distances = distances * pixel_size
        
        return {
            "mean_dist": float(np.mean(distances)),
            "std_dist": float(np.std(distances)),
            "min_dist": float(np.min(distances)),
            "max_dist": float(np.max(distances))
        }


# ============================================================================
# MODULE 5: MESH GENERATION
# ============================================================================
class MeshGenerator:
    """Generate surface meshes from point clouds."""
    
    @staticmethod
    def generate(coords: np.ndarray, depth: int = 9,
                 normal_radius: float = 1000,
                 normal_max_nn: int = 100) -> o3d.geometry.TriangleMesh:
        """
        Create mesh using Poisson surface reconstruction.
        Returns:
            o3d.geometry.TriangleMesh: Generated mesh
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        # Adjust parameters for small point clouds
        n_points = len(coords)
        if n_points < normal_max_nn:
            normal_max_nn = max(min(n_points - 1, 30), 1)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=normal_max_nn
            )
        )
        pcd.orient_normals_consistent_tangent_plane(normal_max_nn)
        
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )

        # Remove low-density vertices (noise)
        densities = np.asarray(densities)
        if len(densities) > 0:
            density_threshold = np.quantile(densities, 0.02)
            vertices_to_keep = densities > density_threshold
            mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        mesh.compute_vertex_normals()
        return mesh


# ============================================================================
# MODULE 6: FEATURE EXTRACTION
# ============================================================================
class FeatureExtractor:
    """Extract refined features from processed data for training."""

    @staticmethod
    def extract(coords: np.ndarray, mesh: o3d.geometry.TriangleMesh,
                neighbor_stats: Dict[str, float]) -> Dict[str, float]:
        """
        Extract final training features only.
        
        Returns:
            Dict with ONLY training features:
                - mean_dist, std_dist, min_dist, max_dist (from neighbor_stats)
                - num_points: point count
                - aspect_ratio: shape descriptor
                - surface_area: envelope size
                - density: derived (num_points / surface_area)
                - bbox_volume: derived (coord_range_x * y * z)
                - class_label: added later in map_processor
        """
        features = {}

        # ===== CORE DISTANCE FEATURES (from neighbor analysis) =====
        features["mean_dist"] = neighbor_stats["mean_dist"]
        features["std_dist"] = neighbor_stats["std_dist"]
        features["min_dist"] = neighbor_stats["min_dist"]
        features["max_dist"] = neighbor_stats["max_dist"]

        # ===== POINT COUNT =====
        features["num_points"] = len(coords)

        # ===== ASPECT RATIO =====
        bbox = mesh.get_axis_aligned_bounding_box()
        extents = bbox.get_extent()
        features["aspect_ratio"] = float(
            max(extents) / min(extents) if min(extents) > 0 else 0
        )

        # ===== SURFACE AREA =====
        try:
            features["surface_area"] = float(mesh.get_surface_area())
        except Exception:
            features["surface_area"] = 0.0
            print("[WARN] Failed to compute surface area, using 0.0")

        # ===== DERIVED FEATURE: DENSITY =====
        if features["surface_area"] > 0:
            features["density"] = features["num_points"] / features["surface_area"]
        else:
            features["density"] = 0.0

        # ===== DERIVED FEATURE: BOUNDING BOX VOLUME =====
        if coords.size > 0:
            coord_range_x = float(coords[:, 0].max() - coords[:, 0].min())
            coord_range_y = float(coords[:, 1].max() - coords[:, 1].min())
            coord_range_z = float(coords[:, 2].max() - coords[:, 2].min())
            features["bbox_volume"] = coord_range_x * coord_range_y * coord_range_z
        else:
            features["bbox_volume"] = 0.0

        return features