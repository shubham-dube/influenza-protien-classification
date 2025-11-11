import joblib
import pandas as pd
import numpy as np
import io

# --- CONFIGURATION ---
MODEL_FILE = 'influenza_protein_classifier.pkl'
# !!! CHANGE THIS INDEX to test a different row from the dataset below !!!
TEST_INDEX = 9
# Note: Data has 22 rows, so index can be from 0 to 21.
# ---------------------

# --- EMBEDDED DATA FOR TESTING (First 22 rows of your original dataset) ---
RAW_TEST_DATA = """mean_dist,std_dist,max_dist,min_dist,num_points,aspect_ratio,surface_area,density,bbox_volume,class_label
0.4851202063690216,0.14937144683286094,2.274270448088794,0.2737899880409241,109799,1.7585201447214813,533333.483769723,0.20587306692974827,11303184.250103435,0
0.4870281909134395,0.1493563208714311,2.724175985062912,0.2737899880409241,108635,1.5982374931283245,527439.256528876,0.20596684576520993,13127698.697164465,0
0.4842306313674288,0.14959526586889663,2.274270448088794,0.2737899880409241,111616,1.6456377265045274,555012.94160704,0.20110522049596874,11279983.954023583,0
0.5284739106476374,0.20878371244089144,1.3689499402046204,0.2737899880409241,575,1.3741772205314535,6055.938800795768,0.094948119344344,64745.01231586341,0
0.532158371059659,0.2120231284989923,1.6877548361521808,0.2737899880409241,829,1.2192243748336797,11483.548371408191,0.072190230161267,149812.7646086506,0
0.531455904884287,0.19562389306675748,1.6427399282455444,0.2737899880409241,959,1.4934822582569802,8799.33281363277,0.10898553564359167,77352.30499181348,0
0.4261533369103226,0.15025170407096444,3.164215248474196,0.2664749950170517,16298,1.1425097977385932,105031.13614316242,0.15517303343063005,3881368.0422611088,0
0.4222919338263703,0.1355889364764261,1.884262759932087,0.2664749950170517,22728,2.131207401754157,81302.43453022365,0.27954882447648005,3132864.0900323014,0
0.42554787559290264,0.15394638084488516,4.410946031589883,0.2664749950170517,14668,1.1961831922621897,105047.3340012193,0.13963229185644777,4415686.240286664,0
0.3517350698629991,0.10774098841438338,2.627705154409431,0.21526999402046204,117078,1.301052493400881,338383.84108500724,0.3459916987306383,6756026.699909092,0
0.3500783760077196,0.10723260879269696,2.328501004528807,0.21526999402046204,120846,1.3595716204673374,363795.0069442126,0.3321815794424346,6491918.976925393,0
0.34945016770257076,0.10582569094332042,1.1592643958310471,0.21526999402046204,120865,1.5033946533739557,331906.29425413965,0.36415398590619685,6766165.020170258,0
0.3432663051377962,0.08227983302802773,1.1371443640906533,0.24243999302387237,10214,1.261879339427125,18854.495063276492,0.541727580914863,433467.0253654041,1
0.33785299641820676,0.08154023389712124,2.806445817957947,0.24243999302387237,9333,1.267136871654199,27274.10973010885,0.34219265421877265,556771.5551372671,1
0.34602299724130026,0.08349486650804096,0.9996056991114345,0.24243999302387237,7977,1.2895822035157267,21403.01335195853,0.3727045285083676,387024.1297905394,1
0.35081497674728107,0.08689752873363704,0.9996056991114345,0.24243999302387237,8114,1.2988497751571537,22790.996367065203,0.35601778304547377,344068.2755786594,1
0.347002893138718,0.08345021828403118,0.8040824910365653,0.24243999302387237,11617,1.4811892458594644,20539.743920132725,0.5655864087289425,580578.338874171,1
0.3499032916726973,0.08598616451968791,0.8741298260707037,0.24243999302387237,7887,1.2150708930680825,21234.8077303178,0.37141847951556484,349815.49415453663,1
0.32504958050261346,0.06572766632777177,1.2362062553040238,0.24243999302387237,9826,1.2758907495217033,21852.373290508112,0.44965367694263497,404584.2082746836,1
0.3252184116140799,0.06464631046837611,0.8398367714079823,0.24243999302387237,11354,1.4317610260711986,18154.43883979093,0.6254117849742776,441652.36328281317,1
0.32463634010355163,0.06611070411369417,0.727319979071617,0.24243999302387237,8204,1.3616124041041973,16486.892807852088,0.4976074082372115,294463.00498078344,1
0.22303408680786588,0.06381274044077807,0.8704449552036801,0.15152501744031904,27269,1.3040058410945972,40154.74884079934,0.6790977602204117,881571.1956528258,1
"""
# --------------------------------------------------------------------------

def load_model(filename):
    """Loads the serialized model pipeline from a .pkl file."""
    try:
        model = joblib.load(filename)
        print(f"✅ Model '{filename}' loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file '{filename}' not found.")
        print("Please ensure the .pkl file is in the same directory as this script.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def get_data_for_test(index):
    """
    Loads the embedded dataset, selects the row by index, and separates features/label.
    """
    df_full = pd.read_csv(io.StringIO(RAW_TEST_DATA))
    
    if index < 0 or index >= len(df_full):
        print(f"❌ Error: Index {index} is out of bounds. Must be between 0 and {len(df_full) - 1}.")
        return None, None

    # Select the row and convert to a DataFrame (required format for the model)
    df_row = df_full.iloc[[index]].copy()
    
    # Separate the true label for comparison
    true_label = df_row['class_label'].iloc[0]
    
    # Features for the classifier
    X_features = df_row.drop('class_label', axis=1)
    
    return X_features, true_label

def run_prediction(model, data, true_label):
    """Makes a prediction using the loaded model."""
    
    print("\n" + "="*50)
    print(f"| Testing Model on Row Index {TEST_INDEX} | True Label: {true_label} |")
    print("="*50)

    # --- Input Data ---
    print("\n--- Input Data ---")
    # Display the input data without the index
    print(data.to_string(index=False))
    
    # --- Prediction Result ---
    print("\n--- Prediction Result ---")

    # The loaded model is the entire Pipeline, automatically handling preprocessing
    prediction = model.predict(data)[0]
    
    # Interpretation
    if prediction == 0:
        interpretation = "Class 0 (Non-Influenza-Critical Protein)"
    elif prediction == 1:
        interpretation = "Class 1 (Influenza-Critical Protein)"
    else:
        interpretation = "Unknown Class"

    print(f"Predicted Class Label: {prediction}")
    print(f"Interpretation: {interpretation}")
    
    # Comparison
    print(f"\n✅ Prediction is CORRECT." if prediction == true_label else f"❌ Prediction is INCORRECT. (Model predicted {prediction}, expected {true_label})")
    
    # Probability
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(data)[0]
        print(f"Probabilities (Class 0, Class 1): {probabilities.round(4)}")


if __name__ == "__main__":
    
    # 1. Load the model
    classifier_pipeline = load_model(MODEL_FILE)
    
    if classifier_pipeline:
        # 2. Get the specific data point
        X_test, y_true = get_data_for_test(TEST_INDEX)
        
        if X_test is not None:
            # 3. Get the prediction
            run_prediction(classifier_pipeline, X_test, y_true)
