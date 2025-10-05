import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# Load pipeline
# -------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("xgb_exoplanet_pipeline.joblib")

pipeline = load_pipeline()

st.set_page_config(
    page_title="🪐 Exoplanet Detection App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔭 Exoplanet Detection using XGBoost")
st.markdown(
    """
Welcome! Enter the characteristics of a star/planet below to predict whether it is an **exoplanet candidate**.  
Missing values are auto-filled by the model pipeline.
"""
)

# -------------------------------
# Get feature names
# -------------------------------
try:
    feature_names = list(pipeline.feature_names_in_)
except Exception:
    st.error("Could not detect pipeline feature names. Hardcode them if needed.")
    st.stop()

# -------------------------------
# Sidebar: User Inputs
# -------------------------------
st.sidebar.header("⚙️ Input Features")

# Organize features by type (numeric vs categorical)
# For demonstration, we'll auto-detect numeric features (simplest approach)
# You can customize categories if you know them
numeric_features = feature_names  # assume all numeric for now

user_inputs = {}
with st.sidebar.form("input_form"):
    st.subheader("🔢 Enter Key Features")
    for f in numeric_features:
        user_inputs[f] = st.number_input(f, value=0.0, format="%.5f", help=f"Feature: {f}")

    submitted = st.form_submit_button("Predict")

# -------------------------------
# Prepare input dataframe
# -------------------------------
input_df = pd.DataFrame(columns=feature_names)
for f in feature_names:
    input_df[f] = [user_inputs.get(f, np.nan)]

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    with st.spinner("Predicting... 🔍"):
        try:
            pred = pipeline.predict(input_df)[0]
            label = "🌍 Exoplanet Detected" if pred == 1 else "❌ Not an Exoplanet"
            st.success(f"**Prediction:** {label}")

            # Optional: probability / confidence
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(input_df)[0][1]
                st.info(f"Model confidence: {proba*100:.2f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------
# Optional: Feature Importance (if available)
# -------------------------------
st.markdown("---")
st.subheader("📊 Feature Importance")
try:
    import matplotlib.pyplot as plt
    import xgboost as xgb

    # Try to extract booster if pipeline ends with XGBClassifier
    if hasattr(pipeline, "named_steps"):
        model = list(pipeline.named_steps.values())[-1]
    else:
        model = pipeline

    if isinstance(model, xgb.XGBClassifier):
        fig, ax = plt.subplots(figsize=(10, 5))
        xgb.plot_importance(model, ax=ax, max_num_features=10)
        st.pyplot(fig)
except Exception:
    st.info("Feature importance plot unavailable for this pipeline.")

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    """
---
Developed for Space Apps Challenge 🌌  
"""
)
