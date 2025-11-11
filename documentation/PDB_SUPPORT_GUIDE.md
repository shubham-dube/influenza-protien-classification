# PDB/ENT File Support Guide

## Overview
Your pipeline now supports both **density maps** (.map, .mrc) and **atomic structure files** (.pdb, .ent, .cif). All files are processed through the same feature extraction pipeline, producing identical output features.

## Key Changes

### 1. **New Directory Structure**
```
data_directories/
├── maps/          # Density maps (.map, .mrc)
├── pdb/           # Atomic structures (.pdb, .ent, .cif)
├── output/        # Results
└── .cache/        # Processing cache
```

### 2. **Automatic File Type Detection**
The pipeline automatically detects file types based on extension:
- `.map`, `.mrc` → Density map processing
- `.pdb`, `.ent`, `.cif` → Atomic structure processing

### 3. **New Dependencies**
Install the updated requirements:
```bash
pip install -r requirements.txt
```

The key new dependency is **Biopython** for PDB parsing.

## Usage Examples

### Process All Files (Default)
```bash
python run_pipeline.py
```
Processes both density maps and PDB files.

### Process Only Density Maps
```bash
python run_pipeline.py --maps-only
```

### Process Only PDB Files
```bash
python run_pipeline.py --pdb-only
```

### Clear Cache and Reprocess Everything
```bash
python run_pipeline.py --clear-cache
```

## File Naming Convention

The pipeline extracts protein type from filenames. Ensure your files follow these patterns:

### Density Maps
- `emd_0025_HA.map`
- `sample_NA.mrc`
- `HA_protein.map`

### PDB Files
- `1abc_HA.pdb`
- `protein_NA.ent`
- `HA_structure.pdb`

The key is to include the protein type (HA, NA, etc.) somewhere in the filename.

## Processing Workflow

### For Density Maps (.map, .mrc):
1. Extract coordinates from density grid (threshold-based)
2. Downsample for efficiency
3. Clean point cloud → Cluster → Extract features

### For PDB Files (.pdb, .ent, .cif):
1. Parse atomic coordinates from PDB structure
2. Extract all atoms (or backbone only, configurable)
3. Clean point cloud → Cluster → Extract features

Both workflows produce **identical feature sets**, allowing combined training.

## Configuration Options

Edit `pipeline_config.py` to customize PDB processing:

```python
PROCESSING_PARAMS = {
    # PDB-specific options
    "backbone_only": False,      # True = only CA, C, N, O atoms
    "atom_types": None,          # List of specific atoms, e.g., ['CA', 'CB']
    
    # Other parameters remain the same...
}
```

### Examples:
- **Backbone only**: Set `"backbone_only": True`
- **Specific atoms**: Set `"atom_types": ['CA', 'CB', 'N']`
- **All atoms** (default): Keep `"backbone_only": False` and `"atom_types": None`

## Output

The pipeline produces the **same output format** regardless of input file type:

### training_data.csv
Contains 10 columns:
1. `mean_dist` - Mean nearest neighbor distance
2. `std_dist` - Standard deviation of distances
3. `max_dist` - Maximum distance
4. `min_dist` - Minimum distance
5. `num_points` - Number of points
6. `aspect_ratio` - Shape descriptor
7. `surface_area` - Envelope surface area
8. `density` - Points per unit surface area
9. `bbox_volume` - Bounding box volume
10. `class_label` - Target classification label

## Troubleshooting

### "ModuleNotFoundError: No module named 'Bio'"
```bash
pip install biopython
```

### PDB File Not Processing
- Check that the file is valid PDB/ENT format
- Ensure protein type (HA/NA) is in the filename
- Check for parse errors in the console output

### Different Features Between Map and PDB
This is expected! Maps and PDB files represent different data:
- **Maps**: Electron density (volume data)
- **PDB**: Atomic positions (discrete points)

The pipeline normalizes both into comparable feature spaces, but magnitudes may differ.

## Advanced: Mixed Dataset Training

Your pipeline now creates a **unified dataset** combining:
- Features from density maps
- Features from atomic structures

This allows you to:
1. Train on diverse data sources
2. Leverage both experimental and computational structures
3. Build more robust classification models

The `class_label` column identifies the protein type, regardless of source file format.

## Example Workflow

```bash
# 1. Place files in appropriate directories
cp *.map data_directories/maps/
cp *.pdb data_directories/pdb/

# 2. Run pipeline
python run_pipeline.py

# 3. Check output
cat data_directories/output/dataset_summary.txt

# 4. Use for training
python train_model.py  # (your training script)
```

## Performance Notes

- **PDB files** typically process faster (fewer points)
- **Density maps** are more computationally intensive
- Use `--skip-cache` to force reprocessing
- Cache tracks file changes automatically (MD5 hashing)

## Support for Additional Formats

The pipeline can be extended to support other formats by:
1. Adding a new converter class in `processing_modules.py`
2. Registering the format in `UnifiedConverter.process()`
3. Adding the extension to `SUPPORTED_EXTENSIONS` in `pipeline_config.py`