# Quick Start Guide

Get your cryo-EM processing pipeline running in 5 minutes!

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## 🚀 Installation

### Step 1: Download the Pipeline

```bash
# If using git
git clone <repository-url>
cd protein_classification

# Or download and extract the ZIP file
```

### Step 2: Run Setup

```bash
python setup.py
```

This will:
- ✅ Check your Python version
- ✅ Install required packages
- ✅ Create directory structure
- ✅ Validate configuration

### Step 3: Add Your Data

Copy your cryo-EM map files to the `maps/` directory:

```bash
cp /path/to/your/*.map maps/
# or
cp /path/to/your/*.mrc maps/
```

**Supported naming patterns:**
- `emd_0025_HA.map` ✓
- `NA_sample.mrc` ✓
- `sample_HA_001.map` ✓

### Step 4: Run Pipeline

```bash
python run_pipeline.py
```

**That's it!** The pipeline will automatically:
1. Process all maps in `maps/` directory
2. Skip already processed maps
3. Generate `output/final_training_dataset.csv`
4. Create summary statistics

---

## 📁 Expected Folder Structure After Setup

```
protein_classification/
├── maps/                    # ← Put your .map/.mrc files here
│   ├── emd_0025_HA.map
│   └── emd_46043_NA.map
├── output/                  # ← Results appear here
│   ├── final_training_dataset.csv
│   └── dataset_summary.txt
├── .cache/                  # ← Auto-generated (don't modify)
├── run_pipeline.py          # ← Main script
├── setup.py                 # ← Setup script
└── ... (other files)
```

---

## 🎯 Common Use Cases

### Process New Maps

Just add new files to `maps/` and run:
```bash
python run_pipeline.py
```
Already processed maps will be skipped automatically!

### Reprocess Everything

```bash
python run_pipeline.py --clear-cache
```

### Analyze Results

```bash
python utils.py output/final_training_dataset.csv summary
```

### Split Train/Test Sets

```bash
python utils.py output/final_training_dataset.csv split
```

---

## 🔧 Customization

### Change Processing Parameters

Edit `pipeline_config.py`:

```python
PROCESSING_PARAMS = {
    "threshold": 0.1,      # Lower = more points extracted
    "downsample": 4,       # Higher = faster but less detail
    "n_clusters": 3,       # Virions per protein
    # ... more options
}
```

### Add New Protein Types

Edit `pipeline_config.py`:

```python
PROTEIN_CLASSES = {
    "HA": 0,
    "NA": 1,
    "M1": 2,     # Add your new type
}
```

---

## 📊 Understanding Output

### Final Dataset Structure

Each row = one cluster from one map

| Column | Example | Description |
|--------|---------|-------------|
| `mean_dist` | 5.234 | Average neighbor distance |
| `num_points` | 1523 | Points in cluster |
| `aspect_ratio` | 1.85 | Shape elongation |
| `protein_type` | "HA" | Detected protein |
| `class_label` | 0 | Numeric class |
| `map_file` | "emd_0025_HA.map" | Source file |

### Example Data Sample

```csv
mean_dist,std_dist,num_points,aspect_ratio,protein_type,class_label,map_file
5.23,2.14,1523,1.85,HA,0,emd_0025_HA.map
4.98,1.89,1645,1.92,HA,0,emd_0025_HA.map
3.45,1.23,2341,1.67,NA,1,emd_46043_NA.map
```

---

## ❓ Troubleshooting

### "No maps found"
**Problem:** Pipeline can't find map files  
**Solution:** Ensure files are in `maps/` with `.map` or `.mrc` extension

### "Module not found"
**Problem:** Missing Python packages  
**Solution:** Run `pip install -r requirements.txt`

### "Out of memory"
**Problem:** Large maps consuming too much RAM  
**Solution:** In `pipeline_config.py`, increase `downsample` to 8 or 16

### Files not being processed
**Problem:** Maps are cached  
**Solution:** Use `python run_pipeline.py --clear-cache`

---

## 💡 Tips

1. **Start Small**: Test with 1-2 maps first
2. **Monitor Progress**: Pipeline shows detailed progress for each map
3. **Check Summary**: Look at `output/dataset_summary.txt` after processing
4. **Backup Cache**: The `.cache/` folder saves processing time - don't delete it!
5. **Version Control**: Add `output/` and `.cache/` to `.gitignore`

---

## 📚 Next Steps

After generating your dataset:

1. **Analyze**: `python utils.py output/final_training_dataset.csv summary`
2. **Visualize**: Load CSV in pandas/Excel to explore features
3. **Train Models**: Use the dataset for machine learning
4. **Iterate**: Adjust parameters and reprocess if needed

---

## 🆘 Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review `dataset_summary.txt` for clues
3. Try processing one map individually
4. Check file permissions on `maps/` directory
5. Verify map files are valid (not corrupted)

---

## ✅ Checklist

- [ ] Python 3.7+ installed
- [ ] Ran `python setup.py`
- [ ] Map files in `maps/` directory
- [ ] Ran `python run_pipeline.py`
- [ ] Found `final_training_dataset.csv` in `output/`
- [ ] Checked `dataset_summary.txt`

**All checked?** Congratulations! 🎉 You're ready to use your dataset!

---

## 📞 Support

For detailed documentation, see [README.md](README.md)

For issues or questions:
- Check existing documentation first
- Review error messages
- Contact: [your-email]