import React, { useState, useEffect } from 'react';
import { Upload, FileText, Activity, Box, Database, BarChart3, Download, AlertCircle, CheckCircle, Loader } from 'lucide-react';

const API_URL = 'http://localhost:5000';

const CryoEMDashboard = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedVirion, setSelectedVirion] = useState(0);
  const [activeTab, setActiveTab] = useState('overview');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const ext = selectedFile.name.split('.').pop().toLowerCase();
      if (['map', 'mrc', 'pdb', 'ent', 'cif'].includes(ext)) {
        setFile(selectedFile);
        setError(null);
      } else {
        setError('Unsupported file format. Please upload .map, .mrc, .pdb, .ent, or .cif files.');
        setFile(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const result = await response.json();
      setAnalysisResult(result);
      setSelectedVirion(0);
    } catch (err) {
      setError(err.message || 'Failed to upload and analyze file');
    } finally {
      setUploading(false);
    }
  };

  const downloadJSON = () => {
    if (!analysisResult) return;
    window.open(`${API_URL}/api/download/${analysisResult.analysis_id}`, '_blank');
  };

  const downloadCSV = () => {
    if (!analysisResult) return;
    window.open(`${API_URL}/api/export_csv/${analysisResult.analysis_id}`, '_blank');
  };

  const UploadSection = () => (
    <div className="bg-white rounded-lg shadow-lg p-8 mb-6">
      <div className="flex items-center mb-6">
        <Upload className="w-8 h-8 text-blue-600 mr-3" />
        <h2 className="text-2xl font-bold text-gray-800">Upload Cryo-EM Data</h2>
      </div>
      
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
        <input
          type="file"
          onChange={handleFileChange}
          accept=".map,.mrc,.pdb,.ent,.cif"
          className="hidden"
          id="file-upload"
        />
        <label htmlFor="file-upload" className="cursor-pointer">
          <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p className="text-lg font-medium text-gray-700 mb-2">
            {file ? file.name : 'Click to select file'}
          </p>
          <p className="text-sm text-gray-500">
            Supported: .map, .mrc, .pdb, .ent, .cif (max 500MB)
          </p>
        </label>
      </div>

      {file && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          className="mt-6 w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
        >
          {uploading ? (
            <>
              <Loader className="w-5 h-5 mr-2 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Activity className="w-5 h-5 mr-2" />
              Analyze File
            </>
          )}
        </button>
      )}

      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
          <AlertCircle className="w-5 h-5 text-red-600 mr-2 flex-shrink-0 mt-0.5" />
          <p className="text-red-700">{error}</p>
        </div>
      )}
    </div>
  );

  const FileInfoCard = () => {
    if (!analysisResult) return null;
    const info = analysisResult.file_info;

    return (
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <FileText className="w-6 h-6 mr-2 text-blue-600" />
          File Information
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-600">Filename</p>
            <p className="font-medium text-gray-800">{info.filename}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Type</p>
            <p className="font-medium text-gray-800">{info.file_type}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Size</p>
            <p className="font-medium text-gray-800">{info.size_mb} MB</p>
          </div>
          {info.map_shape && (
            <div>
              <p className="text-sm text-gray-600">Dimensions</p>
              <p className="font-medium text-gray-800">{info.map_shape.join(' × ')}</p>
            </div>
          )}
          {info.atom_count && (
            <div>
              <p className="text-sm text-gray-600">Atoms</p>
              <p className="font-medium text-gray-800">{info.atom_count.toLocaleString()}</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  const ProcessingStatsCard = () => {
    if (!analysisResult) return null;
    const stats = analysisResult.processing_stats;

    return (
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <Activity className="w-6 h-6 mr-2 text-green-600" />
          Processing Statistics
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 rounded p-4">
            <p className="text-sm text-gray-600">Points Extracted</p>
            <p className="text-2xl font-bold text-blue-600">{stats.total_points_extracted.toLocaleString()}</p>
          </div>
          <div className="bg-green-50 rounded p-4">
            <p className="text-sm text-gray-600">After Cleaning</p>
            <p className="text-2xl font-bold text-green-600">{stats.points_after_cleaning.toLocaleString()}</p>
          </div>
          <div className="bg-purple-50 rounded p-4">
            <p className="text-sm text-gray-600">Virions Detected</p>
            <p className="text-2xl font-bold text-purple-600">{stats.num_virions}</p>
          </div>
          <div className="bg-orange-50 rounded p-4">
            <p className="text-sm text-gray-600">Points Removed</p>
            <p className="text-2xl font-bold text-orange-600">{stats.points_removed.toLocaleString()}</p>
          </div>
        </div>
      </div>
    );
  };

  const VirionSelector = () => {
    if (!analysisResult) return null;

    return (
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Select Virion</h3>
        <div className="flex gap-2">
          {analysisResult.virions.map((virion, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedVirion(idx)}
              className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${
                selectedVirion === idx
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Virion {virion.virion_id}
            </button>
          ))}
        </div>
      </div>
    );
  };

  const VirionDetails = () => {
    if (!analysisResult) return null;
    const virion = analysisResult.virions[selectedVirion];

    return (
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <Box className="w-6 h-6 mr-2 text-purple-600" />
          Virion {virion.virion_id} - Details
        </h3>

        {/* Classification */}
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
          <h4 className="font-bold text-gray-800 mb-2 flex items-center">
            <CheckCircle className="w-5 h-5 mr-2 text-green-600" />
            Classification Result
          </h4>
          <p className="text-2xl font-bold text-purple-700 mb-2">
            {virion.classification.predicted_label}
          </p>
          <p className="text-sm text-gray-600">
            Confidence: <span className="font-bold">{(virion.classification.confidence * 100).toFixed(1)}%</span>
          </p>
          <div className="mt-3">
            {Object.entries(virion.classification.probabilities || {}).map(([cls, prob]) => (
              <div key={cls} className="mb-2">
                <div className="flex justify-between text-sm mb-1">
                  <span>{cls}</span>
                  <span className="font-medium">{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Features */}
        <h4 className="font-bold text-gray-800 mb-3">Extracted Features</h4>
        <div className="grid grid-cols-3 gap-3">
          {Object.entries(virion.features).map(([key, value]) => (
            <div key={key} className="bg-gray-50 rounded p-3">
              <p className="text-xs text-gray-600 mb-1">{key.replace(/_/g, ' ').toUpperCase()}</p>
              <p className="font-bold text-gray-800">
                {typeof value === 'number' ? value.toFixed(4) : value}
              </p>
            </div>
          ))}
        </div>

        {/* Statistics */}
        <h4 className="font-bold text-gray-800 mt-6 mb-3">Spatial Statistics</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Centroid (X, Y, Z):</span>
            <span className="font-medium">
              {virion.statistics.centroid.map(v => v.toFixed(2)).join(', ')}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Std Dev (X, Y, Z):</span>
            <span className="font-medium">
              {virion.statistics.std_dev.map(v => v.toFixed(2)).join(', ')}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">X Range:</span>
            <span className="font-medium">
              [{virion.statistics.coord_range.x.map(v => v.toFixed(2)).join(' to ')}]
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Y Range:</span>
            <span className="font-medium">
              [{virion.statistics.coord_range.y.map(v => v.toFixed(2)).join(' to ')}]
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Z Range:</span>
            <span className="font-medium">
              [{virion.statistics.coord_range.z.map(v => v.toFixed(2)).join(' to ')}]
            </span>
          </div>
        </div>
      </div>
    );
  };

  const VisualizationSection = () => {
    if (!analysisResult) return null;
    const viz = analysisResult.visualizations;

    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
            <BarChart3 className="w-6 h-6 mr-2 text-orange-600" />
            Density Distribution
          </h3>
          <img 
            src={`data:image/png;base64,${viz.density_histogram}`}
            alt="Density Histogram"
            className="w-full rounded"
          />
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Cluster Projections</h3>
          <img 
            src={`data:image/png;base64,${viz.cluster_projections}`}
            alt="Cluster Projections"
            className="w-full rounded"
          />
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Feature Comparison</h3>
          <img 
            src={`data:image/png;base64,${viz.feature_comparison}`}
            alt="Feature Comparison"
            className="w-full rounded"
          />
        </div>
      </div>
    );
  };

  const Model3DSection = () => {
    if (!analysisResult) return null;
    const model = analysisResult.model_3d[selectedVirion];

    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <Database className="w-6 h-6 mr-2 text-indigo-600" />
          3D Point Cloud Data
        </h3>
        <div className="bg-gray-50 rounded p-4 mb-4">
          <p className="text-sm text-gray-600 mb-2">
            Original Points: <span className="font-bold">{model.original_points.toLocaleString()}</span>
          </p>
          <p className="text-sm text-gray-600 mb-2">
            Displayed Points: <span className="font-bold">{model.displayed_points.toLocaleString()}</span>
          </p>
          <p className="text-sm text-gray-600">
            Downsampling helps with web visualization performance
          </p>
        </div>
        <div className="bg-blue-50 rounded p-4">
          <p className="text-xs text-gray-600 mb-2">Bounds (Angstroms):</p>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <p className="font-bold">Min</p>
              <p>{model.bounds.min.map(v => v.toFixed(1)).join(', ')}</p>
            </div>
            <div>
              <p className="font-bold">Max</p>
              <p>{model.bounds.max.map(v => v.toFixed(1)).join(', ')}</p>
            </div>
            <div>
              <p className="font-bold">Center</p>
              <p>{model.bounds.center.map(v => v.toFixed(1)).join(', ')}</p>
            </div>
          </div>
        </div>
        <p className="text-sm text-gray-500 mt-4 italic">
          Note: Integrate Three.js or similar library for interactive 3D visualization
        </p>
      </div>
    );
  };

  const DownloadSection = () => {
    if (!analysisResult) return null;

    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <Download className="w-6 h-6 mr-2 text-green-600" />
          Export Results
        </h3>
        <div className="flex gap-4">
          <button
            onClick={downloadJSON}
            className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center justify-center"
          >
            <Download className="w-5 h-5 mr-2" />
            Download JSON
          </button>
          <button
            onClick={downloadCSV}
            className="flex-1 bg-green-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-green-700 transition-colors flex items-center justify-center"
          >
            <Download className="w-5 h-5 mr-2" />
            Download CSV
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Cryo-EM Analysis Dashboard
          </h1>
          <p className="text-gray-600">
            Upload and analyze cryo-EM density maps and PDB structures with ML-powered classification
          </p>
        </div>

        {/* Upload Section */}
        <UploadSection />

        {/* Results Section */}
        {analysisResult && (
          <>
            {/* Tabs */}
            <div className="bg-white rounded-lg shadow-lg mb-6">
              <div className="flex border-b">
                {['overview', 'virions', 'visualizations', '3d-models'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`flex-1 py-4 px-6 font-medium transition-colors ${
                      activeTab === tab
                        ? 'border-b-2 border-blue-600 text-blue-600'
                        : 'text-gray-600 hover:text-gray-800'
                    }`}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1).replace('-', ' ')}
                  </button>
                ))}
              </div>
            </div>

            {/* Tab Content */}
            {activeTab === 'overview' && (
              <>
                <FileInfoCard />
                <ProcessingStatsCard />
                <DownloadSection />
              </>
            )}

            {activeTab === 'virions' && (
              <>
                <VirionSelector />
                <VirionDetails />
              </>
            )}

            {activeTab === 'visualizations' && <VisualizationSection />}

            {activeTab === '3d-models' && (
              <>
                <VirionSelector />
                <Model3DSection />
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default CryoEMDashboard;