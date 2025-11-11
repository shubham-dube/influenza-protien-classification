Deployment Guide: Influenza Protein Classifier

This guide explains how to use the exported machine learning model (influenza_protein_classifier.pkl) to make predictions on new, unseen 3D protein data.

The model exported is a complete scikit-learn Pipeline, meaning it contains all the necessary preprocessing steps (feature scaling and transformation) and the final trained classifier (Logistic Regression, Random Forest, or SVM). You only need to provide the raw input data; the pipeline handles the rest automatically.

1. Prerequisites

To use the model, you must have a Python environment with the following libraries installed:

pip install scikit-learn pandas joblib


scikit-learn: The machine learning library used to train the model.

pandas: Used to structure the input data correctly (as a DataFrame).

joblib: Used to load the .pkl file efficiently.

2. Model File Location

Ensure that the influenza_protein_classifier.pkl file is located in the same directory as the Python script (test_classifier.py) or your main application file.

3. Preparing Input Data

The most crucial step is formatting your new data correctly. The model expects exactly 9 features in a specific order:

mean_dist

std_dist

max_dist

min_dist

num_points

aspect_ratio

surface_area

density

bbox_volume

Your new data must be provided as a Pandas DataFrame with these exact column names.

4. Testing the Model (Using test_classifier.py)

The provided Python script test_classifier.py shows the exact process:

Step A: Save the Script

Save the test_classifier.py code into a file in the same directory as your .pkl model.

Step B: Run the Test

Execute the script from your terminal:

python test_classifier.py


Expected Output

The script will load the model, use the sample data point, and output the prediction:

✅ Model 'influenza_protein_classifier.pkl' loaded successfully.

--- Input Data ---
 mean_dist  std_dist  max_dist  min_dist  num_points  aspect_ratio  surface_area  density  bbox_volume
    0.4501    0.1205     2.512     0.250       15000          1.45       45000.0     0.55    1200000.0

--- Prediction Result ---
Predicted Class Label: 1
Interpretation: Class 1 (Example: Influenza-Critical Protein)
Probabilities (Class 0, Class 1): [0.2245 0.7755]
