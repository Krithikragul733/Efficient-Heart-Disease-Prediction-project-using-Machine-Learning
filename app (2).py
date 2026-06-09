
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
# StandardScaler is specifically imported because it's part of the pipeline logic for LR model
# and needs to be loaded if it was saved, or if it's used in transformation as done here.
from sklearn.preprocessing import StandardScaler

# --- Configuration and Constants ---
BEST_MODEL_NAME = 'Random Forest' # Hardcoded from previous analysis
MODEL_FILENAME = f'{BEST_MODEL_NAME.replace(" ", "_").lower()}_best_model.joblib'
SCALER_FILENAME = 'scaler.joblib'
FEATURE_NAMES_FILENAME = 'feature_names.joblib'
TOP_N_FEATURES_FILENAME = 'top_n_features.joblib'

# --- Load Saved Assets ---
# Load the best model
try:
    loaded_model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_FILENAME}' not found. Please ensure it has been saved in the same directory as app.py.")
    st.stop()

# Load the scaler
try:
    scaler = joblib.load(SCALER_FILENAME)
except FileNotFoundError:
    st.error(f"Error: Scaler file '{SCALER_FILENAME}' not found. Please ensure it has been saved in the same directory as app.py.")
    st.stop()

# Load feature names
try:
    feature_names = joblib.load(FEATURE_NAMES_FILENAME)
except FileNotFoundError:
    st.error(f"Error: Feature names file '{FEATURE_NAMES_FILENAME}' not found. Please ensure it has been saved in the same directory as app.py.")
    st.stop()

# Load top N features for explanation
try:
    top_n_features_dict = joblib.load(TOP_N_FEATURES_FILENAME)
    top_n_features_series = pd.Series(top_n_features_dict)
except FileNotFoundError:
    st.warning(f"Top N features file '{TOP_N_FEATURES_FILENAME}' not found. Explanations might be limited.")
    top_n_features_series = pd.Series({}) # Empty series if not found

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction (XAI)",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("❤️ Explainable Ensemble Machine Learning Framework for Heart Disease Prediction")
st.markdown("--- ✨ ***Powered by AI for Health*** ✨ ---")
st.write(
    "This application predicts the likelihood of heart disease based on various patient parameters. "
    "It utilizes an explainable ensemble model to provide insightful predictions. "
    "Simply adjust the patient's health parameters below and click 'Predict Heart Disease'."
)

st.header("Patient Health Parameters")

# --- User Input Fields ---
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 29, 77, 50)
    sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    cp = st.selectbox('Chest Pain Type (CP)', [0, 1, 2, 3])
    trestbps = st.slider('Resting Blood Pressure (trestbps) mm/Hg', 94, 200, 120)
    chol = st.slider('Serum Cholestoral (chol) mg/dl', 126, 564, 230)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1], format_func=lambda x: 'True' if x == 1 else 'False')
    restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', [0, 1, 2])

with col2:
    thalach = st.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 150)
    exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    oldpeak = st.slider('ST depression induced by exercise relative to rest (oldpeak)', 0.0, 6.2, 1.0, 0.1)
    slope = st.selectbox('Slope of the peak exercise ST segment (slope)', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (ca)', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])

# --- Create Input DataFrame ---
user_input = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

input_df = pd.DataFrame([user_input], columns=feature_names) # Use loaded feature_names

st.subheader("Review Patient Input")
st.dataframe(input_df)

# --- Prediction Button and Results ---
st.subheader("Get Prediction")

if st.button("Predict Heart Disease"):
    # Prepare the input data for the model
    # If the best model was Logistic Regression, it expects scaled data.
    # Using BEST_MODEL_NAME constant here.
    if BEST_MODEL_NAME == 'Logistic Regression':
        processed_input = scaler.transform(input_df)
    else:
        processed_input = input_df

    # Make prediction
    prediction = loaded_model.predict(processed_input)

    st.write("### Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ **Heart Disease Detected**")
    else:
        st.success("✅ **No Heart Disease Detected**")

    # --- Prediction Probability ---
    st.subheader("Prediction Probability")
    if hasattr(loaded_model, 'predict_proba'):
        probabilities = loaded_model.predict_proba(processed_input)[0]
        st.write(f"Probability of No Heart Disease: **{probabilities[0]*100:.2f}%**")
        st.write(f"Probability of Heart Disease: **{probabilities[1]*100:.2f}%**")
        st.progress(probabilities[1])
    else:
        st.info("Model does not support probability prediction.")

    # --- Optional Explainability ---
    st.subheader("Factors Influencing Prediction")
    st.info(
        "This section provides a simplified view of features that significantly influence the prediction. "
        "For a comprehensive explanation, refer to the SHAP analysis in Phase 4 of the notebook."
    )

    if not top_n_features_series.empty:
        st.write("Top important features (general impact):")
        for feature, importance in top_n_features_series.items():
            st.markdown(f"- **{feature}**: (Importance: {importance:.4f})")
        st.write(
            "*Note: The actual impact for this specific patient may vary. High values of 'thalach' (max heart rate) or 'cp' (chest pain type) often indicate a higher risk or protective factor depending on their specific values.*"
        )
    else:
        st.warning("Feature importance data not available for detailed explanation.")

# --- Healthcare Disclaimer ---
st.markdown("--- ✨ ***Important Disclaimer*** ✨ ---")
st.warning(
    "**Disclaimer:** This system is for educational purposes only and is not a replacement "
    "for professional medical diagnosis, advice, or treatment. Always consult with a qualified "
    "healthcare professional for any health concerns or before making any decisions related to your health."
    "The predictions provided here are based on a machine learning model and should not be used as the sole basis for medical decisions."
)
