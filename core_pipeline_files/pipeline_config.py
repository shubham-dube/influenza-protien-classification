#!/usr/bin/env python3
"""
pipeline_config.py
-----------------
Central configuration for the automated cryo-EM processing pipeline.
Supports both density maps and PDB files (with realistic map conversion).

KEY UPDATE: Added PDB-to-realistic-map conversion parameters to ensure
biologically meaningful feature extraction from atomic structures.
"""

import os

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) 
MAPS_DIR = os.path.join(ROOT_DIR, "data_directories/maps")
PDB_DIR = os.path.join(ROOT_DIR, "data_directories/pdb")
CACHE_DIR = os.path.join(ROOT_DIR, "data_directories/.cache")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data_directories/output")
FINAL_DATASET = os.path.join(OUTPUT_DIR, "training_data.csv")

# PDB map conversion cache (subdirectory of CACHE_DIR)
PDB_MAP_CACHE_DIR = os.path.join(CACHE_DIR, "pdb_maps")

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================
PROCESSING_PARAMS = {
    # Map to coordinate conversion (for .map/.mrc files)
    "threshold": 0.1,           # Density threshold for coordinate extraction
    "downsample": 4,            # Keep every Nth voxel
    
    # PDB to realistic map conversion (for .pdb/.ent/.cif files)
    # These parameters control the simulated cryo-EM map generation
    "apix": 1.2,                # Voxel size (Angstroms/pixel) for generated maps
    "resolution": 6.0,          # Target resolution (Angstroms) - realistic range: 3-12
    "box_size": 256,            # Box dimensions (voxels) - typical EMDB: 200-400
    "min_separation": 150.0,    # Minimum virion separation (Angstroms) - ensures clean clustering
    "noise_level": 2.0,         # Noise strength (0-5) - 2.0 is realistic for EMDB
    
    # CTF simulation (if EMAN2 available)
    "defocus_min": 15000,       # Minimum defocus (Angstroms)
    "defocus_max": 25000,       # Maximum defocus (Angstroms)
    
    # Point cloud cleaning
    "nb_neighbors": 200,        # For statistical outlier removal
    "std_ratio": 0.1,          # Standard deviation ratio for outliers
    
    # Clustering
    "n_clusters": 3,           # Number of virions per protein (matches PDB conversion)
    
    # Nearest neighbors
    "k_neighbors": 3,          # Number of nearest neighbors to compute
    "kth_ignore": 1,           # Ignore first k self-neighbors
    "pixel_size": 0.209,       # Default pixel size in nm (for maps)
    
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
# Supported file extensions
SUPPORTED_EXTENSIONS = ['.map', '.mrc', '.pdb', '.ent', '.cif']

# Expected naming patterns
EXPECTED_MAP_PATTERNS = [
    "*_HA.map", "*_HA.mrc", "*_HA.pdb", "*_HA.ent",
    "*_NA.map", "*_NA.mrc", "*_NA.pdb", "*_NA.ent",
    "HA_*.map", "HA_*.mrc", "HA_*.pdb", "HA_*.ent",
    "NA_*.map", "NA_*.mrc", "NA_*.pdb", "NA_*.ent",
]

# ============================================================================
# FEATURE COLUMNS (FINAL TRAINING FEATURES ONLY)
# ============================================================================
# Final training features - THESE ARE THE ONLY COLUMNS IN OUTPUT CSV
TRAINING_FEATURE_COLS = [
    "mean_dist", 
    "std_dist", 
    "max_dist",
    "min_dist",
    "num_points",
    "aspect_ratio", 
    "surface_area",
    "density",      # Derived: num_points / surface_area
    "bbox_volume",  # Derived: coord_range_x * coord_range_y * coord_range_z
    "class_label"
]

# ============================================================================
# CACHE SETTINGS
# ============================================================================
CACHE_INDEX_FILE = os.path.join(CACHE_DIR, "processed_maps.json")
ENABLE_CACHING = True  # Set to False to reprocess all maps

# ============================================================================
# PDB-TO-MAP CONVERSION NOTES
# ============================================================================
"""
PDB Conversion Pipeline:
-----------------------
For PDB/ENT/CIF files, the pipeline automatically:

1. Creates 3 spatially separated copies (virions) with:
   - Separation ≥ 150 Å (clean clustering)
   - Random rotations (orientation diversity)

2. Generates realistic cryo-EM density map:
   - Resolution filtering (3-12 Å range)
   - Gaussian noise (Poisson-like distribution)
   - CTF effects (if EMAN2 available)
   - Realistic voxel size (1.0-1.5 Å)

3. Saves cached map as:
   .cache/pdb_maps/<hash>.mrc
   
4. Extracts features from map (NOT from PDB directly)
   - This ensures biologically meaningful features
   - Features match EMDB density map processing

Why this matters:
-----------------
Direct feature extraction from PDB atomic coordinates gives garbage values
that have no biological meaning for cryo-EM analysis. The conversion step
simulates realistic experimental conditions.

EMAN2 vs Fallback:
-----------------
- WITH EMAN2: Full pipeline with CTF, realistic noise, projections
- WITHOUT EMAN2: Simplified Gaussian density + noise (less realistic)

To install EMAN2: https://blake.bcm.edu/emanwiki/EMAN2
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [MAPS_DIR, PDB_DIR, CACHE_DIR, OUTPUT_DIR, PDB_MAP_CACHE_DIR]:
        os.makedirs(directory, exist_ok=True)

def get_protein_type(filename: str) -> str:
    """
    Extract protein type from filename.
    
    Examples:
        emd_0025_HA.map -> HA
        NA_sample.mrc -> NA
        1abc_HA.pdb -> HA
    """
    filename_upper = filename.upper()
    for protein in PROTEIN_CLASSES.keys():
        if protein in filename_upper:
            return protein
    return "UNKNOWN"

def get_class_label(protein_type: str) -> int:
    """Get numeric class label for protein type."""
    return PROTEIN_CLASSES.get(protein_type, -1)

def extract_map_id(filename: str) -> str:
    """
    Extract map ID from filename.
    
    Examples:
        emd_0025_HA.map -> emd_0025
        NA_sample_123.mrc -> NA_sample_123
        1abc_HA.pdb -> 1abc
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Try to extract EMD ID pattern
    import re
    emd_match = re.search(r'emd_\d+', name, re.IGNORECASE)
    if emd_match:
        return emd_match.group(0).lower()
    
    # Try to extract PDB ID pattern (4 characters at start)
    pdb_match = re.search(r'^[0-9][a-zA-Z0-9]{3}', name)
    if pdb_match:
        return pdb_match.group(0).lower()
    
    # Otherwise return the full filename without extension
    return name

def is_supported_file(filename: str) -> bool:
    """Check if file extension is supported."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS

def get_conversion_info():
    """Get information about PDB-to-map conversion settings."""
    return {
        'enabled': True,
        'cache_dir': PDB_MAP_CACHE_DIR,
        'apix': PROCESSING_PARAMS['apix'],
        'resolution': PROCESSING_PARAMS['resolution'],
        'box_size': PROCESSING_PARAMS['box_size'],
        'n_virions': PROCESSING_PARAMS['n_clusters'],
        'min_separation': PROCESSING_PARAMS['min_separation'],
        'noise_level': PROCESSING_PARAMS['noise_level'],
    }