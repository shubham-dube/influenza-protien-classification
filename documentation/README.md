# Automated Cryo-EM Processing Pipeline

A modular, automated pipeline for processing multiple cryo-EM density maps and generating machine learning training datasets.

## Features

✅ **Fully Automated**: Process multiple maps with a single command  
✅ **Smart Caching**: Skip already processed maps automatically  
✅ **Modular Design**: Easy to extend and customize  
✅ **No Intermediate Files**: Only generates final dataset  
✅ **Progress Tracking**: Detailed logging and status updates  
✅ **Error Handling**: Continues processing even if individual maps fail  

---

## Folder Structure

```
protein_classification/
├── maps/                          # Place your .map/.mrc files here
│   ├── emd_0025_HA.map
│   ├── emd_46043_NA.map
│   └── ... (more maps)
├── output/                        # Generated outputs
│   ├── final_training_dataset.csv # Main output
│   └── dataset_summary.txt        # Summary statistics
├── .cache/                        # Processing cache (auto-generated)
│   └── processed_maps.json
├── run_pipeline.py               # Main execution script
├── pipeline_config.py            # Configuration
├── processing_modules.py         # Core processing functions
├── map_processor.py              # Single map processor
├── cache_manager.py              # Cache management
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Installation

### 1. Clone or Download Repository

```bash
git clone <repository-url>
cd protein_classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- numpy
- pandas
- mrcfile
- open3d
- scikit-learn

---

## Quick Start

### 1. Add Your Map Files

Place all `.map` or `.mrc` files in the `maps/` directory:

```bash
cp /path/to/your/maps/*.map maps/
```

### 2. Run the Pipeline

```bash
python run_pipeline.py
```

That's it! The pipeline will:
- Automatically detect all map files
- Process each map (skip already processed ones)
- Extract features from each cluster
- Append to the final dataset
- Generate summary statistics

---

## Usage Examples

### Basic Usage

```bash
# Process all new maps in ./maps/
python run_pipeline.py
```

### Advanced Options

```bash
# Reprocess all maps (ignore cache)
python run_pipeline.py --clear-cache

# Process without using cache
python run_pipeline.py --skip-cache

# Use custom directories
python run_pipeline.py --maps-dir /path/to/maps --output-dir /path/to/output
```

---

## Configuration

Edit `pipeline_config.py` to customize processing parameters:

```python
PROCESSING_PARAMS = {
    "threshold": 0.1,           # Density threshold
    "downsample": 4,            # Downsampling factor
    "nb_neighbors": 200,        # Outlier removal
    "std_ratio": 0.1,          # Outlier threshold
    "n_clusters": 3,           # Virions per protein
    "k_neighbors": 3,          # Nearest neighbors
    "pixel_size": 0.209,       # Pixel size (nm)
    "poisson_depth": 9,        # Mesh quality
}
```

### Adding New Protein Types

```python
PROTEIN_CLASSES = {
    "HA": 0,
    "NA": 1,
    "M1": 2,  # Add new types here
    "NP": 3,
}
```

---

## Output Format

### Final Dataset (`output/final_training_dataset.csv`)

Each row represents one cluster from one map:

| Column | Description |
|--------|-------------|
| `mean_dist` | Mean nearest neighbor distance |
| `std_dist` | Standard deviation of distances |
| `min_dist` | Minimum distance |
| `max_dist` | Maximum distance |
| `num_points` | Number of points in cluster |
| `aspect_ratio` | Mesh aspect ratio |
| `volume` | Mesh volume |
| `surface_area` | Mesh surface area |
| `coord_range_x/y/z` | Coordinate ranges |
| `map_file` | Source map filename |
| `protein_type` | Protein type (HA/NA/etc) |
| `class_label` | Numeric class label |
| `cluster_id` | Cluster ID within map |

---

## Pipeline Steps

For each map file, the pipeline:

1. **Converts** density map to 3D coordinates
2. **Cleans** point cloud (removes outliers)
3. **Clusters** points into virions (KMeans)
4. **Computes** nearest neighbor statistics
5. **Generates** surface mesh (Poisson)
6. **Extracts** geometric features
7. **Appends** to final dataset

---

## Caching System

The pipeline uses smart caching to avoid reprocessing:

- **Automatic**: Maps are cached after successful processing
- **Hash-based**: Detects if map files have changed
- **Persistent**: Cache survives between runs
- **Manual control**: Use `--clear-cache` to reset

### Cache Location

`.cache/processed_maps.json`

---

## File Naming Convention

The pipeline automatically detects protein types from filenames:

✅ **Supported patterns:**
- `*_HA.map` → Hemagglutinin
- `*_NA.mrc` → Neuraminidase
- `HA_*.map` → Hemagglutinin
- `NA_*.map` → Neuraminidase

Example filenames:
- `emd_0025_HA.map` ✓
- `sample_NA.mrc` ✓
- `HA_variant_1.map` ✓

---

## Extending the Pipeline

### Add Custom Features

Edit `processing_modules.py`, class `FeatureExtractor`:

```python
def extract(self, coords, mesh, neighbor_stats):
    features = neighbor_stats.copy()
    
    # Add your custom feature here
    features["custom_metric"] = compute_custom_metric(coords)
    
    return features
```

### Add Processing Step

Create new class in `processing_modules.py`:

```python
class CustomAnalyzer:
    @staticmethod
    def analyze(coords):
        # Your analysis code
        return result
```

Then add to `map_processor.py`:

```python
self.custom_analyzer = CustomAnalyzer()
# ... use in process() method
```

---

## Troubleshooting

### No maps found
```
[ERROR] No .map or .mrc files found in ./maps/
```
**Solution**: Place your map files in the `maps/` directory

### Import errors
```
ModuleNotFoundError: No module named 'mrcfile'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Open3D visualization issues
The pipeline runs in headless mode (no GUI required). If you see warnings about visualization, they can be safely ignored.

### Memory issues
For large maps, adjust parameters in `pipeline_config.py`:
- Increase `downsample` (e.g., 8 or 16)
- Increase `threshold` (e.g., 0.2)

---

## Performance Tips

1. **Parallel Processing**: For many maps, split into batches and run multiple instances
2. **Downsampling**: Increase `downsample` parameter for faster processing
3. **Clustering**: Reduce `n_clusters` if you have fewer virions per map
4. **Cache**: Keep cache enabled to avoid reprocessing

---

## License

[Your License Here]

## Citation

If you use this pipeline in your research, please cite:

```
[Your Citation Here]
```

---

## Contact

For issues, questions, or contributions:
- Email: [your-email]
- GitHub: [repository-url]