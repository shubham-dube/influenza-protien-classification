#!/usr/bin/env python3
"""
pipeline_config.py
-----------------
Central configuration for the automated cryo-EM processing pipeline.
"""

import os

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) 
MAPS_DIR = os.path.join(ROOT_DIR, "data_directories/maps")
CACHE_DIR = os.path.join(ROOT_DIR, "data_directories/.cache")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data_directories/output")
FINAL_DATASET = os.path.join(OUTPUT_DIR, "final_training_dataset.csv")

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================
PROCESSING_PARAMS = {
    # Map to coordinate conversion
    "threshold": 0.1,           # Density threshold for coordinate extraction
    "downsample": 4,            # Keep every Nth voxel
    
    # Point cloud cleaning
    "nb_neighbors": 200,        # For statistical outlier removal
    "std_ratio": 0.1,          # Standard deviation ratio for outliers
    
    # Clustering
    "n_clusters": 3,           # Number of virions per protein
    
    # Nearest neighbors
    "k_neighbors": 3,          # Number of nearest neighbors to compute
    "kth_ignore": 1,           # Ignore first k self-neighbors
    "pixel_size": 0.209,       # Default pixel size in nm
    
    # Mesh reconstruction
    "poisson_depth": 9,        # Depth for Poisson reconstruction
    "normal_radius": 1000,     # Radius for normal estimation
    "normal_max_nn": 100,      # Max neighbors for normal estimation
}

# ============================================================================
# PROTEIN CLASS MAPPING
# ============================================================================
PROTEIN_CLASSES = {
    "HA": 0,
    "NA": 1,
    # Add more protein types here as needed
    # "M1": 2,
    # "NP": 3,
}

# ============================================================================
# FILE NAMING CONVENTIONS
# ============================================================================
EXPECTED_MAP_PATTERNS = [
    "*_HA.map", "*_HA.mrc",
    "*_NA.map", "*_NA.mrc",
    "HA_*.map", "HA_*.mrc",
    "NA_*.map", "NA_*.mrc",
]

# ============================================================================
# FEATURE COLUMNS
# ============================================================================
COORDINATE_COLS = ["x", "y", "z"]
SUMMARY_FEATURE_COLS = [
    "mean_dist", 
    "std_dist", 
    "aspect_ratio", 
    "num_points",
    "protein_type",
    "class_label"
]

# ============================================================================
# CACHE SETTINGS
# ============================================================================
CACHE_INDEX_FILE = os.path.join(CACHE_DIR, "processed_maps.json")
ENABLE_CACHING = True  # Set to False to reprocess all maps

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [MAPS_DIR, CACHE_DIR, OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)

def get_protein_type(filename: str) -> str:
    """
    Extract protein type from filename.
    
    Examples:
        emd_0025_HA.map -> HA
        NA_sample.mrc -> NA
    """
    filename_upper = filename.upper()
    for protein in PROTEIN_CLASSES.keys():
        if protein in filename_upper:
            return protein
    return "UNKNOWN"

def get_class_label(protein_type: str) -> int:
    """Get numeric class label for protein type."""
    return PROTEIN_CLASSES.get(protein_type, -1)