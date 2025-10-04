import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer 

# --- CONFIGURATION & ARTIFACT LOADING ---
# NOTE: The setup_model_files.py script MUST be run before this app to create these files.

try:
    with open('gbc_model.pkl', 'rb') as f:
        gbc_model = pickle.load(f)
    
    with open('imputer.pkl', 'rb') as f:
        # Load the fitted Imputer
        imputer = pickle.load(f)
        
    with open('feature_cols.pkl', 'rb') as f:
        # Load the full list of features for the ML model (in the correct order)
        FULL_FEATURE_COLS = pickle.load(f)

    with open('imputer_cols.pkl', 'rb') as f:
        # Load the subset of columns that the imputer was trained on
        IMPUTATION_FEATURE_COLS = pickle.load(f)

    MODEL_LOADED = True
except FileNotFoundError as e:
    st.error(f"Model files not found: {e.filename}. Please run 'setup_model_files.py' first.")
    FULL_FEATURE_COLS = []
    IMPUTATION_FEATURE_COLS = []
    MODEL_LOADED = False

# Recreate the label encoder (based on the known classes)
le = LabelEncoder()
le.fit(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
class_names = le.classes_ 


# --- PREDICTION FUNCTION ---

def predict_exoplanet(input_data_series: pd.Series):
    """
    Takes raw user input, preprocesses it, and returns the model prediction.
    """
    if not MODEL_LOADED:
        return "ERROR", 0.0, [0.0, 0.0, 0.0]

    # 1. Convert Series to DataFrame and ensure correct feature order
    # This DataFrame contains ALL features, including the binary flags.
    input_df = pd.DataFrame([input_data_series], columns=FULL_FEATURE_COLS)
    
    # 2. Imputation: Apply transformation only to the subset of numerical columns
    
    # Isolate columns that need imputation (This is the fix for the ValueError)
    df_to_impute = input_df[IMPUTATION_FEATURE_COLS]
    
    # Transform these columns (imputer only sees columns it was fitted on)
    imputed_array = imputer.transform(df_to_impute)
    
    # Reintegrate imputed values back into the main DataFrame structure
    input_df[IMPUTATION_FEATURE_COLS] = imputed_array
    
    # Ensure the final DataFrame passed to the model has all columns in the correct order
    processed_df = input_df[FULL_FEATURE_COLS]
    
    # 3. Make Prediction
    
    # Make prediction (returns index 0, 1, or 2)
    prediction_index = gbc_model.predict(processed_df)[0]
    
    # Get confidence scores (probabilities for each class)
    probabilities = gbc_model.predict_proba(processed_df)[0]
    
    predicted_class = class_names[prediction_index]
    confidence = probabilities[prediction_index]
    
    return predicted_class, confidence, probabilities


# --- STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Exoplanet AI Classification Dashboard", layout="wide")

st.title("🔭 NASA Exoplanet AI Classifier")
st.markdown("A Gradient Boosting model trained on Kepler data to classify observations into Confirmed, Candidate, or False Positive.")

# --- Tab 1: Predict New Data ---
st.header("1. Classify New Observation")

# Input Form for a single data point
with st.form("exoplanet_input_form"):
    st.subheader("Input Physical Parameters for Classification")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Planetary & Orbit**")
        # Ensure initial values are reasonable for a planet
        koi_period = st.number_input("Orbital Period (days)", value=10.0, min_value=0.1, format="%.4f")
        koi_duration = st.number_input("Transit Duration (hrs)", value=3.0, min_value=0.1, format="%.4f")
        koi_prad = st.number_input("Planet Radius (Earth Radii)", value=2.5, min_value=0.1, format="%.2f")
        koi_impact = st.number_input("Impact Parameter", value=0.5, min_value=0.0, max_value=1.5, format="%.3f")
        koi_depth = st.number_input("Transit Depth (ppm)", value=500.0, min_value=1.0, format="%.1f")
        
    with col2:
        st.markdown("**Stellar & Environment**")
        koi_steff = st.number_input("Stellar Effective Temp (K)", value=5800.0, min_value=2000.0, format="%.1f")
        koi_slogg = st.number_input("Stellar Surface Gravity (logg)", value=4.5, min_value=0.0, max_value=5.0, format="%.3f")
        koi_srad = st.number_input("Stellar Radius (Solar Radii)", value=1.0, min_value=0.1, format="%.3f")
        koi_teq = st.number_input("Equilibrium Temp (K)", value=800.0, min_value=100.0, format="%.1f")
        koi_insol = st.number_input("Insolation Flux", value=100.0, min_value=0.1, format="%.2f")
        
    with col3:
        st.markdown("**Model Indicators & Flags**")
        koi_model_snr = st.number_input("Model SNR", value=50.0, min_value=6.0, format="%.1f")
        koi_score = st.number_input("Disposition Score (0.0 - 1.0)", value=0.99, min_value=0.0, max_value=1.0, format="%.4f")
        
        st.markdown("*False Positive Flags (Set to 1)*")
        # Boolean inputs for binary flags
        koi_fpflag_co = st.checkbox("Centroid Offset", value=False)
        koi_fpflag_nt = st.checkbox("Not Transit-Like", value=False)
        koi_fpflag_ss = st.checkbox("Stellar Eclipse", value=False)
        koi_fpflag_ec = st.checkbox("Ephemeris Contamination", value=False)
        
        # Fixed, less critical parameters for manual input (included in FULL_FEATURE_COLS)
        koi_time0bk = 170.0 
        koi_kepmag = 15.0


    submitted = st.form_submit_button("CLASSIFY CANDIDATE")

    if submitted and MODEL_LOADED:
        
        # 1. Collect inputs into a Pandas Series
        input_data = pd.Series({
            'koi_period': koi_period, 'koi_time0bk': koi_time0bk, 'koi_impact': koi_impact,
            'koi_duration': koi_duration, 'koi_depth': koi_depth, 'koi_prad': koi_prad,
            'koi_teq': koi_teq, 'koi_insol': koi_insol, 'koi_model_snr': koi_model_snr,
            'koi_steff': koi_steff, 'koi_slogg': koi_slogg, 'koi_srad': koi_srad,
            'koi_kepmag': koi_kepmag, 'koi_score': koi_score,
            'koi_fpflag_nt': int(koi_fpflag_nt), 'koi_fpflag_ss': int(koi_fpflag_ss),
            'koi_fpflag_co': int(koi_fpflag_co), 'koi_fpflag_ec': int(koi_fpflag_ec)
        })
        
        # 2. Get prediction
        predicted_class, confidence, probabilities = predict_exoplanet(input_data)

        st.subheader("Classification Result:")
        
        # Display main result with appropriate color
        if predicted_class == 'CONFIRMED':
            st.success(f"**Predicted Disposition:** {predicted_class} 🎉 (Confidence: {confidence:.2%})")
        elif predicted_class == 'CANDIDATE':
            st.warning(f"**Predicted Disposition:** {predicted_class} 🟠 (Confidence: {confidence:.2%})")
        else:
            st.error(f"**Predicted Disposition:** {predicted_class} ❌ (Confidence: {confidence:.2%})")
            
        st.markdown("---")
        
        # Display probabilities for full transparency
        st.write("**Probability Distribution:**")
        
        # Map probabilities back to class names
        prob_dict = {class_names[i]: probabilities[i] for i in range(len(class_names))}
        prob_df = pd.DataFrame(prob_dict, index=['Probability']).T.sort_values(
            by='Probability', ascending=False
        )
        
        st.dataframe(prob_df, use_container_width=True)

st.divider()

# --- Tab 2: Model Insights ---
st.header("2. Model Insights & Explainability")
st.markdown("Understanding *why* the model makes a decision is crucial. Here are the most influential features. ")

# Check if the plot file exists before trying to display it
if os.path.exists('feature_importance_plot.png'):
    st.image('feature_importance_plot.png', caption='Top 15 Most Important Features for Exoplanet Classification (by GBC Model)')
else:
    st.warning("Feature importance plot image not found. Please run 'setup_model_files.py' to generate the plot.")

st.divider()

# --- Tab 3: Contribution and Metrics ---
st.header("3. Project Metrics & Contribution")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("Trained Model Performance (Test Set)")
    st.markdown("""
        The model achieved high performance, especially in filtering out false positives.
        
        * **Overall Accuracy:** $92.73\%$
        * **False Positive Precision:** $99.6\%$ (Very reliable at rejecting false alarms)
        * **Confirmed Recall:** $89.6\%$ (Good at finding true positives)
    """)

with col_m2:
    st.subheader("Continuous Learning")
    st.info("The architecture supports periodic retraining to incorporate new, verified data from researchers, ensuring the model remains up-to-date and robust over time.")
    st.button("Submit Verified Data (Conceptual Feature) 🔄")
