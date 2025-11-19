#!/usr/bin/env python3
"""
api_server.py
-------------
Flask API server for cryo-EM map and PDB file analysis.
Provides comprehensive analysis, visualization, and classification.

Endpoints:
    POST /api/upload           - Upload and analyze file
    GET  /api/analysis/{id}    - Get analysis results
    GET  /api/visualization/{id}/{type} - Get visualizations
    GET  /api/model3d/{id}/{virion} - Get 3D model data
    GET  /api/health           - Health check
"""

import os
import sys
import json
import uuid
import shutil
import tempfile
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import mrcfile
from Bio.PDB import PDBParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64

# Import your processing modules
from core_pipeline_files.map_processor import MapProcessor
from core_pipeline_files.pipeline_config import PROCESSING_PARAMS, get_protein_type, get_class_label
from core_pipeline_files.processing_modules import UnifiedConverter, PointCloudCleaner, VirionClusterer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='cryoem_uploads_')
ANALYSIS_FOLDER = tempfile.mkdtemp(prefix='cryoem_analysis_')
MODEL_PATH = 'influenza_protein_classifier.pkl'
ALLOWED_EXTENSIONS = {'map', 'mrc', 'pdb', 'ent', 'cif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANALYSIS_FOLDER'] = ANALYSIS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Global storage for analysis results
analysis_cache = {}

# Load ML model
try:
    ml_model = joblib.load(MODEL_PATH)
    print(f"✓ Loaded model: {MODEL_PATH}")
except Exception as e:
    print(f"⚠ Warning: Could not load model: {e}")
    ml_model = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_info(filepath: str) -> Dict:
    """Extract basic file information."""
    stat = os.stat(filepath)
    ext = filepath.rsplit('.', 1)[1].lower()
    
    info = {
        'filename': os.path.basename(filepath),
        'extension': ext,
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'file_type': 'Density Map' if ext in ['map', 'mrc'] else 'Atomic Structure',
        'uploaded_at': datetime.now().isoformat()
    }
    
    # Try to extract more specific info
    try:
        if ext in ['map', 'mrc']:
            with mrcfile.open(filepath, permissive=True) as mrc:
                info['map_shape'] = list(mrc.data.shape)
                if hasattr(mrc, 'voxel_size'):
                    vs = mrc.voxel_size
                    if hasattr(vs, 'x'):
                        info['voxel_size'] = [float(vs.x), float(vs.y), float(vs.z)]
                    else:
                        info['voxel_size'] = [float(vs)] * 3
        elif ext in ['pdb', 'ent']:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', filepath)
            atom_count = sum(1 for _ in structure.get_atoms())
            info['atom_count'] = atom_count
            info['num_chains'] = sum(1 for _ in structure.get_chains())
    except Exception as e:
        print(f"Warning: Could not extract detailed info: {e}")
    
    return info


def create_density_histogram(coords: np.ndarray) -> str:
    """Create histogram of spatial density distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate distances from center
    center = coords.mean(axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    
    ax.hist(distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Distance from Center (Å)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Spatial Distribution of Points', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def create_cluster_visualization(coords: np.ndarray, labels: np.ndarray) -> str:
    """Create 2D projection of clusters."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    projections = [
        (0, 1, 'X-Y'),
        (0, 2, 'X-Z'),
        (1, 2, 'Y-Z')
    ]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for ax, (dim1, dim2, title) in zip(axes, projections):
        for i in np.unique(labels):
            cluster_coords = coords[labels == i]
            ax.scatter(cluster_coords[:, dim1], cluster_coords[:, dim2], 
                      c=colors[i % len(colors)], label=f'Virion {i+1}', 
                      alpha=0.6, s=1)
        
        ax.set_xlabel(f'{title[0]} (Å)', fontsize=10)
        ax.set_ylabel(f'{title[-1]} (Å)', fontsize=10)
        ax.set_title(f'{title} Projection', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def create_feature_comparison(virion_features: List[Dict]) -> str:
    """Create bar chart comparing features across virions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    feature_sets = [
        (['mean_dist', 'std_dist'], 'Distance Statistics (nm)'),
        (['num_points'], 'Point Count'),
        (['surface_area', 'bbox_volume'], 'Size Metrics'),
        (['density', 'aspect_ratio'], 'Shape Descriptors')
    ]
    
    virion_ids = [f"Virion {i+1}" for i in range(len(virion_features))]
    
    for ax, (features, title) in zip(axes, feature_sets):
        x = np.arange(len(virion_ids))
        width = 0.8 / len(features)
        
        for i, feat in enumerate(features):
            values = [v[feat] for v in virion_features]
            ax.bar(x + i * width, values, width, label=feat, alpha=0.8)
        
        ax.set_xlabel('Virion', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x + width * (len(features) - 1) / 2)
        ax.set_xticklabels(virion_ids)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def classify_protein_features(features: Dict) -> Dict:
    """Classify protein using ML model."""
    if ml_model is None:
        return {'error': 'Model not loaded'}
    
    feature_order = [
        'mean_dist', 'std_dist', 'max_dist', 'min_dist', 'num_points', 
        'aspect_ratio', 'surface_area', 'density', 'bbox_volume'
    ]
    
    input_df = pd.DataFrame([features], columns=feature_order)
    
    try:
        prediction = ml_model.predict(input_df)[0]
        probabilities = ml_model.predict_proba(input_df)[0]
        
        class_names = {0: 'HA (Hemagglutinin)', 1: 'NA (Neuraminidase)'}
        
        return {
            'predicted_class': int(prediction),
            'predicted_label': class_names.get(prediction, f'Class {prediction}'),
            'confidence': float(max(probabilities)),
            'probabilities': {
                class_names.get(i, f'Class {i}'): float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
    except Exception as e:
        return {'error': str(e)}


def get_3d_model_data(coords: np.ndarray, downsample: int = 10) -> Dict:
    """Prepare 3D point cloud data for web visualization."""
    # Downsample for web display
    coords_downsampled = coords[::downsample]
    
    # Normalize to [0, 1] range for better visualization
    coords_normalized = (coords_downsampled - coords_downsampled.min(axis=0)) / \
                       (coords_downsampled.max(axis=0) - coords_downsampled.min(axis=0))
    
    return {
        'points': coords_normalized.tolist(),
        'original_points': len(coords),
        'displayed_points': len(coords_downsampled),
        'bounds': {
            'min': coords.min(axis=0).tolist(),
            'max': coords.max(axis=0).tolist(),
            'center': coords.mean(axis=0).tolist()
        }
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': ml_model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload and analyze a cryo-EM map or PDB file.
    
    Returns comprehensive analysis including:
    - File metadata
    - Processing statistics
    - Per-virion features
    - ML classification results
    - Visualization references
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported format. Allowed: {ALLOWED_EXTENSIONS}'}), 400
    
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{analysis_id}_{filename}")
        file.save(filepath)
        
        print(f"\n{'='*70}")
        print(f"Processing upload: {filename}")
        print(f"Analysis ID: {analysis_id}")
        print(f"{'='*70}")
        
        # Extract file info
        file_info = get_file_info(filepath)
        
        # Process the file
        processor = MapProcessor(params=PROCESSING_PARAMS)
        
        # Get detailed processing results
        print("[1/5] Converting to coordinates...")
        coords = processor.converter.process(
            filepath,
            threshold=PROCESSING_PARAMS["threshold"],
            downsample=PROCESSING_PARAMS["downsample"]
        )
        
        print("[2/5] Cleaning point cloud...")
        cleaned_coords = processor.cleaner.clean(
            coords,
            nb_neighbors=PROCESSING_PARAMS["nb_neighbors"],
            std_ratio=PROCESSING_PARAMS["std_ratio"]
        )
        
        print("[3/5] Clustering virions...")
        clusters, labels = processor.clusterer.cluster(
            cleaned_coords,
            n_clusters=PROCESSING_PARAMS["n_clusters"]
        )
        
        print("[4/5] Extracting features...")
        virion_data = []
        for i, cluster_coords in enumerate(clusters):
            # Compute all features
            neighbor_stats = processor.neighbor_analyzer.compute_distances(
                cluster_coords,
                k=PROCESSING_PARAMS["k_neighbors"],
                kth_ignore=PROCESSING_PARAMS["kth_ignore"],
                pixel_size=PROCESSING_PARAMS["pixel_size"]
            )
            
            mesh = processor.mesh_generator.generate(
                cluster_coords,
                depth=PROCESSING_PARAMS["poisson_depth"],
                normal_radius=PROCESSING_PARAMS["normal_radius"],
                normal_max_nn=PROCESSING_PARAMS["normal_max_nn"]
            )
            
            features = processor.feature_extractor.extract(
                cluster_coords,
                mesh,
                neighbor_stats
            )
            
            # Classify this virion
            classification = classify_protein_features(features)
            
            # Additional statistics
            stats = {
                'virion_id': i + 1,
                'features': features,
                'classification': classification,
                'statistics': {
                    'centroid': cluster_coords.mean(axis=0).tolist(),
                    'std_dev': cluster_coords.std(axis=0).tolist(),
                    'coord_range': {
                        'x': [float(cluster_coords[:, 0].min()), float(cluster_coords[:, 0].max())],
                        'y': [float(cluster_coords[:, 1].min()), float(cluster_coords[:, 1].max())],
                        'z': [float(cluster_coords[:, 2].min()), float(cluster_coords[:, 2].max())]
                    }
                }
            }
            
            virion_data.append(stats)
        
        print("[5/5] Generating visualizations...")
        
        # Create visualizations
        density_hist = create_density_histogram(cleaned_coords)
        cluster_viz = create_cluster_visualization(cleaned_coords, labels)
        feature_comp = create_feature_comparison([v['features'] for v in virion_data])
        
        # Prepare 3D model data for each virion
        virion_3d_data = []
        for i, cluster_coords in enumerate(clusters):
            virion_3d_data.append(get_3d_model_data(cluster_coords, downsample=20))
        
        # Compile complete analysis result
        analysis_result = {
            'analysis_id': analysis_id,
            'file_info': file_info,
            'processing_stats': {
                'total_points_extracted': len(coords),
                'points_after_cleaning': len(cleaned_coords),
                'points_removed': len(coords) - len(cleaned_coords),
                'num_virions': len(clusters),
                'processing_time': datetime.now().isoformat()
            },
            'virions': virion_data,
            'visualizations': {
                'density_histogram': density_hist,
                'cluster_projections': cluster_viz,
                'feature_comparison': feature_comp
            },
            'model_3d': virion_3d_data
        }
        
        # Cache the result
        analysis_cache[analysis_id] = {
            'result': analysis_result,
            'filepath': filepath,
            'coords': cleaned_coords,
            'clusters': clusters,
            'labels': labels
        }
        
        print(f"✓ Analysis complete: {analysis_id}")
        
        return jsonify(analysis_result), 200
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Retrieve cached analysis results."""
    if analysis_id not in analysis_cache:
        return jsonify({'error': 'Analysis not found'}), 404
    
    return jsonify(analysis_cache[analysis_id]['result']), 200


@app.route('/api/model3d/<analysis_id>/<int:virion_id>', methods=['GET'])
def get_3d_model(analysis_id, virion_id):
    """Get 3D point cloud data for a specific virion."""
    if analysis_id not in analysis_cache:
        return jsonify({'error': 'Analysis not found'}), 404
    
    cached = analysis_cache[analysis_id]
    
    if virion_id < 1 or virion_id > len(cached['clusters']):
        return jsonify({'error': 'Invalid virion ID'}), 400
    
    cluster_coords = cached['clusters'][virion_id - 1]
    model_data = get_3d_model_data(cluster_coords, downsample=10)
    
    return jsonify(model_data), 200


@app.route('/api/download/<analysis_id>', methods=['GET'])
def download_results(analysis_id):
    """Download analysis results as JSON."""
    if analysis_id not in analysis_cache:
        return jsonify({'error': 'Analysis not found'}), 404
    
    result = analysis_cache[analysis_id]['result']
    
    # Create temp file
    output_path = os.path.join(app.config['ANALYSIS_FOLDER'], f"{analysis_id}.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return send_file(output_path, as_attachment=True, 
                    download_name=f"analysis_{analysis_id}.json")


@app.route('/api/export_csv/<analysis_id>', methods=['GET'])
def export_csv(analysis_id):
    """Export features as CSV for training."""
    if analysis_id not in analysis_cache:
        return jsonify({'error': 'Analysis not found'}), 404
    
    result = analysis_cache[analysis_id]['result']
    
    # Extract features for all virions
    rows = []
    for virion in result['virions']:
        rows.append(virion['features'])
    
    df = pd.DataFrame(rows)
    
    # Create temp file
    output_path = os.path.join(app.config['ANALYSIS_FOLDER'], f"{analysis_id}.csv")
    df.to_csv(output_path, index=False)
    
    return send_file(output_path, as_attachment=True,
                    download_name=f"features_{analysis_id}.csv")


@app.route('/api/list', methods=['GET'])
def list_analyses():
    """List all cached analyses."""
    analyses = []
    for aid, data in analysis_cache.items():
        analyses.append({
            'analysis_id': aid,
            'filename': data['result']['file_info']['filename'],
            'file_type': data['result']['file_info']['file_type'],
            'num_virions': len(data['result']['virions']),
            'timestamp': data['result']['processing_stats']['processing_time']
        })
    
    return jsonify({'analyses': analyses}), 200


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("CRYO-EM ANALYSIS API SERVER")
    print("="*70)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Analysis folder: {ANALYSIS_FOLDER}")
    print(f"Model loaded: {ml_model is not None}")
    print(f"Supported formats: {ALLOWED_EXTENSIONS}")
    print("="*70)
    print("\nStarting server on http://localhost:5000")
    print("API Docs: http://localhost:5000/api/health")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)