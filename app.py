import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

model = load_model()

# Page config
st.set_page_config(page_title="Horse Rating Predictor", page_icon="🐴", layout="wide")

st.title("🐴 Horse Rating Predictor")
st.markdown("Predict horse racing performance ratings using AI")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Horse Information")
    
    name = st.text_input("Horse Name", value="Secretariat")
    age = st.number_input("Age (years)", min_value=1, max_value=20, value=4)
    sex = st.selectbox("Sex", ["M", "F", "C", "G"], index=0)
    
    st.subheader("Performance Metrics")
    erg = st.slider("ERG", min_value=0, max_value=100, value=75)
    rawErg = st.slider("Raw ERG", min_value=0, max_value=100, value=70)
    ems3 = st.slider("EMS3", min_value=0, max_value=100, value=65)

with col2:
    st.subheader("Economic Data")
    fee = st.number_input("Stud Fee ($)", min_value=0, max_value=500000, value=50000, step=5000)
    crop = st.number_input("Crop Size", min_value=0, max_value=50, value=5)
    
    st.subheader("Pedigree")
    sire = st.text_input("Sire (Father)", value="American Pharoah")
    dam = st.text_input("Dam (Mother)", value="Leslie's Lady")

# Predict button
if st.button("🎯 Predict Rating", type="primary"):
    # Prepare input data (adjust features to match your model)
    input_data = pd.DataFrame({
        'age': [age],
        'erg': [erg],
        'rawErg': [rawErg],
        'ems3': [ems3],
        'fee': [fee],
        'crop': [crop],
        'sex_M': [1 if sex == 'M' else 0],
        'sex_F': [1 if sex == 'F' else 0],
        'sex_C': [1 if sex == 'C' else 0],
        'sex_G': [1 if sex == 'G' else 0],
        # Add other features your model expects
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.success(f"## Predicted Rating: {prediction:.2f}")
    
    # Rating interpretation
    if prediction >= 90:
        st.info("🏆 **Elite Champion** - Exceptional performance expected")
    elif prediction >= 75:
        st.info("⭐ **High Performer** - Strong competitive potential")
    elif prediction >= 60:
        st.info("✅ **Average Performer** - Solid racing prospect")
    else:
        st.info("📊 **Developing** - May need more training or suited for breeding")
    
    # Show input summary
    with st.expander("📋 View Input Details"):
        st.write(input_data)

# Sidebar with info
st.sidebar.title("About")
st.sidebar.info(
    "This app predicts horse racing ratings using an XGBoost machine learning model. "
    "Enter the horse's characteristics to get a predicted performance rating."
)

st.sidebar.title("Model Info")
st.sidebar.metric("Model Type", "XGBoost")
st.sidebar.metric("Training RMSE", "8.12")
st.sidebar.metric("Test RMSE", "9.13")