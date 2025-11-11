import joblib
import pandas as pd

# Load the model once when your application starts
MODEL_PATH = 'influenza_protein_classifier.pkl'
loaded_model = joblib.load(MODEL_PATH)

def classify_protein(feature_values: dict):
    """
    Takes a dictionary of 9 feature values and returns the predicted class label.
    """
    # 1. Define the mandatory feature order
    feature_order = [
        'mean_dist', 'std_dist', 'max_dist', 'min_dist', 'num_points', 
        'aspect_ratio', 'surface_area', 'density', 'bbox_volume'
    ]

    # 2. Convert raw dictionary into the required DataFrame format
    input_df = pd.DataFrame([feature_values], columns=feature_order)

    # 3. Predict. The model handles all internal preprocessing.
    prediction = loaded_model.predict(input_df)[0]
    
    # Optionally get probabilities
    # probabilities = loaded_model.predict_proba(input_df)[0]
    
    return prediction

# Example Usage in your main app logic:
my_new_protein_data = {
    'mean_dist': 0.4501, 
    'std_dist': 0.1205, 
    'max_dist': 2.512, 
    'min_dist': 0.250, 
    'num_points': 15000, 
    'aspect_ratio': 1.45, 
    'surface_area': 45000.0, 
    'density': 0.55, 
    'bbox_volume': 1200000.0
}

result = classify_protein(my_new_protein_data)
print(f"Classification Result: {result}")