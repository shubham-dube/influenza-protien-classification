# PDB to Realistic Cryo-EM Map Conversion

## 🎯 Overview

This pipeline now includes **automatic PDB-to-realistic-map conversion** to ensure biologically meaningful feature extraction from atomic structures.

### Why This Matters

**❌ WRONG:** Extracting features directly from PDB atomic coordinates
- Gives unrealistic, meaningless values
- No density information
- No noise characteristics
- No resolution effects
- Features don't match EMDB maps

**✅ CORRECT:** Convert PDB → Realistic Cryo-EM Map → Extract Features
- Mimics real experimental conditions
- Includes realistic noise and CTF effects
- Features comparable to EMDB density maps
- Biologically meaningful measurements

---

## 🔧 Installation

### Required Dependencies

```bash
# Core dependencies (already in your pipeline)
pip install numpy pandas scipy open3d scikit-learn biopython mrcfile

# Optional but HIGHLY RECOMMENDED: EMAN2
# For realistic CTF simulation and noise modeling
# Installation: https://blake.bcm.edu/emanwiki/EMAN2
```

### With EMAN2 (Recommended)
- Full realistic simulation
- CTF effects
- Realistic noise from real micrographs
- Defocus simulation

### Without EMAN2 (Fallback)
- Simplified Gaussian density
- Basic noise model
- Still functional but less realistic

---

## 📁 Directory Structure

```
project/
├── pipeline_scripts/
│   ├── pdb_to_realistic_map.py      # NEW: PDB→Map converter
│   ├── processing_modules.py        # UPDATED: Integrated conversion
│   ├── map_processor.py             # UPDATED: Handles PDB files
│   ├── pipeline_config.py           # UPDATED: PDB conversion params
│   └── run_pipeline.py              # Works automatically
│
└── data_directories/
    ├── pdb/                         # Place PDB files here
    │   ├── 1abc_HA.pdb
    │   ├── 2def_NA.ent
    │   └── 3ghi_HA.cif
    │
    ├── maps/                        # Place density maps here
    │   ├── emd_0025_HA.map
    │   └── emd_0026_NA.mrc
    │
    └── .cache/
        └── pdb_maps/                # AUTO: Cached converted maps
            ├── abc123def456.mrc     # Cached HA map
            └── fed654cba321.mrc     # Cached NA map
```

---

## 🚀 Usage

### Basic Usage (Automatic)

The pipeline **automatically detects and converts PDB files**:

```bash
# Process all files (maps + PDBs)
python run_pipeline.py

# Process only PDB files
python run_pipeline.py --pdb-only

# Process only density maps
python run_pipeline.py --maps-only

# Clear cache and reprocess everything
python run_pipeline.py --clear-cache
```

### Manual Conversion (Advanced)

If you want to convert PDB files manually:

```python
from pdb_to_realistic_map import convert_pdb_to_map

# Basic conversion
output_map = convert_pdb_to_map(
    'input.pdb',
    output_file='output.mrc'
)

# Custom parameters
output_map = convert_pdb_to_map(
    'input.pdb',
    apix=1.5,              # Voxel size
    resolution=8.0,        # Target resolution (Å)
    box_size=300,          # Box dimensions
    n_virions=3,           # Number of copies
    min_separation=180.0,  # Separation (Å)
    noise_level=2.5        # Noise strength
)
```

---

## ⚙️ Conversion Parameters

### Default Settings (in `pipeline_config.py`)

```python
PROCESSING_PARAMS = {
    # PDB-to-map conversion
    "apix": 1.2,              # Voxel size (Å/pixel)
    "resolution": 6.0,        # Target resolution (Å)
    "box_size": 256,          # Box dimensions (voxels)
    "n_clusters": 3,          # Number of virions (matches clustering)
    "min_separation": 150.0,  # Virion separation (Å)
    "noise_level": 2.0,       # Noise strength (0-5)
    
    # CTF parameters (EMAN2 only)
    "defocus_min": 15000,     # Min defocus (Å)
    "defocus_max": 25000,     # Max defocus (Å)
}
```

### Parameter Tuning Guide

| Parameter | Recommended Range | Effect |
|-----------|-------------------|--------|
| `apix` | 1.0 - 1.5 | Smaller = higher resolution, larger files |
| `resolution` | 3.0 - 12.0 | Match target EMDB maps |
| `box_size` | 200 - 400 | Depends on protein size |
| `min_separation` | 150 - 200 | Ensure clean clustering |
| `noise_level` | 1.5 - 3.0 | 2.0 is typical EMDB |

---

## 🔄 Conversion Pipeline

### Step-by-Step Process

```
PDB File (1abc_HA.pdb)
    ↓
[1] Create 3 Virion Copies
    • Copy 1: Original position
    • Copy 2: +150Å in X
    • Copy 3: +150Å in Y
    ↓
[2] Random Rotations
    • Each virion rotated randomly
    • Mimics real orientation diversity
    ↓
[3] Generate Ideal Density Map
    • Convert atoms → electron density
    • Gaussian atomic potentials
    ↓
[4] Apply Resolution Filter
    • Gaussian filter at target resolution
    • Mimics detector MTF
    ↓
[5] Generate 2D Projections (EMAN2)
    • 1000+ projections at random angles
    • Mimics particle images
    ↓
[6] Apply CTF Effects (EMAN2)
    • Defocus blur
    • Phase flipping
    • Contrast reduction
    ↓
[7] Add Realistic Noise
    • Gaussian + Poisson noise
    • From real micrograph patches (EMAN2)
    ↓
[8] Reconstruct 3D Map
    • Back-projection
    • Missing wedge artifacts
    • Realistic FSC resolution
    ↓
Realistic Cryo-EM Map (.mrc)
    ↓
[9] Extract Coordinates
    • Threshold density
    • Downsample
    ↓
[10] Extract Features
    • Same as density map pipeline
    • Biologically meaningful!
```

---

## 💾 Caching System

### How Caching Works

1. **Cache Key Generation**
   - Hash of: filename + conversion parameters
   - Example: `abc123def456789.mrc`

2. **Cache Lookup**
   - Before conversion, checks cache
   - If found, uses cached map instantly
   - Saves 30-120 seconds per file

3. **Cache Location**
   ```
   data_directories/.cache/pdb_maps/
   ├── abc123def456.mrc  # 1abc_HA.pdb converted
   ├── def789ghi012.mrc  # 2def_NA.pdb converted
   └── ...
   ```

4. **Cache Management**
   ```bash
   # Clear only PDB map cache
   rm -rf data_directories/.cache/pdb_maps/*
   
   # Clear all caches
   python run_pipeline.py --clear-cache
   ```

---

## 📊 Output Comparison

### Features from PDB (Direct) - ❌ WRONG
```
mean_dist:     0.003    # Angstrom-level (meaningless)
num_points:    8234     # Just atom count
surface_area:  423.1    # Van der Waals surface
density:       19.45    # atoms/Å² (not EM density!)
bbox_volume:   8234.2   # Atomic bounding box
```

### Features from Converted Map - ✅ CORRECT
```
mean_dist:     2.134    # Real EM voxel spacing
num_points:    45123    # Density voxels above threshold
surface_area:  15234.5  # Envelope surface
density:       2.96     # points/surface (EM-relevant)
bbox_volume:   123456.7 # Reconstructed volume
```

---

## 🔍 Validation

### How to Verify Conversion Quality

1. **Visual Inspection**
   ```python
   import mrcfile
   import matplotlib.pyplot as plt
   
   # Load converted map
   with mrcfile.open('converted.mrc') as mrc:
       data = mrc.data
       
       # Plot central slice
       plt.imshow(data[data.shape[0]//2], cmap='gray')
       plt.title('Converted Map - Central Slice')
       plt.show()
   ```

2. **Check Virion Separation**
   - Should see 3 distinct density blobs
   - Spaced ≥150Å apart
   - Different orientations

3. **Verify Noise Level**
   - Background should have visible noise
   - Not perfectly smooth
   - Similar to EMDB maps

4. **Compare Features**
   ```python
   import pandas as pd
   
   df = pd.read_csv('training_data.csv')
   
   # Check feature distributions
   print(df.groupby('class_label')['density'].describe())
   print(df.groupby('class_label')['mean_dist'].describe())
   ```

---

## 🐛 Troubleshooting

### Issue: "EMAN2 not found, using fallback"
- **Solution:** Install EMAN2 for best results
- **Workaround:** Fallback method works but is less realistic

### Issue: Cache not working
```bash
# Check cache directory
ls -la data_directories/.cache/pdb_maps/

# Verify permissions
chmod -R 755 data_directories/.cache/
```

### Issue: Conversion too slow
- **Cause:** Large PDB files or small `apix` values
- **Solution:** 
  ```python
  # Increase voxel size
  "apix": 1.5,  # Instead of 1.0
  
  # Reduce box size
  "box_size": 200,  # Instead of 300
  ```

### Issue: Features still look wrong
- **Check:** Verify conversion actually ran
  ```bash
  ls -la data_directories/.cache/pdb_maps/
  ```
- **Debug:** Enable verbose output
  ```bash
  python run_pipeline.py --verbose
  ```

---

## 📈 Performance

### Conversion Time Benchmarks

| Method | First Run | Cached | Quality |
|--------|-----------|--------|---------|
| EMAN2 Full | 60-120s | Instant | ⭐⭐⭐⭐⭐ |
| Fallback | 10-30s | Instant | ⭐⭐⭐ |

### Memory Usage
- EMAN2: ~2-4 GB per conversion
- Fallback: ~500 MB per conversion

### Disk Space
- Each cached map: 50-200 MB
- Recommendation: 5-10 GB free for cache

---

## 🎓 Best Practices

1. **Always use EMAN2 if possible**
   - Significantly more realistic results
   - Better feature quality

2. **Match parameters to your EMDB maps**
   ```python
   # If comparing to EMDB-12345 at 6Å
   "resolution": 6.0,
   "apix": 1.2,  # Check EMDB entry
   ```

3. **Use caching for repeated experiments**
   - Saves 99% of processing time
   - Ensures consistency

4. **Validate on known structures first**
   - Use PDB structures with known EMDB maps
   - Compare feature distributions

5. **Monitor cache size**
   ```bash
   du -sh data_directories/.cache/pdb_maps/
   ```

---

## 📚 References

- **EMAN2 Documentation:** https://blake.bcm.edu/emanwiki/EMAN2
- **Cryo-EM Map Simulation:** [Henderson, 2013]
- **CTF Theory:** [Wade, 1992]
- **EMDB Format:** https://www.emdataresource.org/

---

## ✅ Quick Checklist

Before running your pipeline:

- [ ] PDB files in `data_directories/pdb/`
- [ ] Filenames contain protein type (HA, NA, etc.)
- [ ] EMAN2 installed (optional but recommended)
- [ ] Sufficient disk space (5-10 GB)
- [ ] Cache directory has write permissions
- [ ] Parameters match your target resolution

---

## 🆘 Support

If you encounter issues:

1. Check the conversion log output
2. Verify cache directory exists and is writable
3. Try manual conversion first
4. Compare with known EMDB maps
5. Check parameter ranges

**The pipeline will now automatically handle PDB files correctly, generating realistic cryo-EM-like features that are biologically meaningful and comparable to EMDB density map features!** 🎉