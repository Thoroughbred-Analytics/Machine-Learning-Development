import streamlit as st
import pandas as pd
import numpy as np
import joblib

from dataHandler import *

# Load your trained model
@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

# Loading the model can take some time, so we show a spinner
with st.spinner('Loading model...'):
    model = load_model()

# Loading form to numeric mapping
print(model.get_booster().feature_names)  # Debugging: Check feature names expected by the model

# Page config
st.set_page_config(page_title="Horse Rating Predictor", page_icon="🐴", layout="wide")

st.title("Horse Rating Predictor")
st.markdown("Enter details of a horse to predict its racing performance.")

# Create two columns
col1, col2 = st.columns(2)

with col1:

    st.subheader("Horse Details")

    st.write(" ")
    st.write(" ")
    
    with st.container(border=True):
        st.subheader("Horse Information")
        name = st.text_input("Horse Name", value="Secretariat")
        age = st.number_input("Age (years)", min_value=1, max_value=20, value=4)
        sex = st.selectbox("Sex", ["M", "F", "C", "G"], index=0)
        fee = st.number_input("Stud Fee ($)", min_value=0, max_value=500000, value=50000, step=5000)
        crop = st.number_input("Crop Size", min_value=0, max_value=50, value=5)
    
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")


    with st.container(border=True):
        st.subheader("Performance Metrics")
        form = st.text_input("Form", value="G3p")
        erg = st.number_input("ERG", min_value=0, max_value=100, value=75)
        rawErg = st.number_input("Raw ERG", min_value=0, max_value=100, value=75)
        ems3 = st.number_input("EMS3", min_value=0, max_value=100, value=75)

with col2:

    st.subheader("Parent Statistics")

    with st.container(border=True):
        st.subheader("Sire")

        sireRating = st.number_input("Sire Rating", min_value=0, max_value=100, value=75)
        sireErg = st.number_input("Sire ERG", min_value=0, max_value=100, value=75)
        sireForm = st.text_input("Sire Form", value="G1w")

    with st.container(border=True):
        st.subheader("Dam")

        damRating = st.number_input("Dam Rating", min_value=0, max_value=100, value=75)
        damErg = st.number_input("Dam ERG", min_value=0, max_value=100, value=75)
        damForm = st.text_input("Dam Form", value="G2p")

    with st.container(border=True):
        st.subheader("Sire's Dam")

        bmSireRating = st.number_input("Sire's Dam Rating", min_value=0, max_value=100, value=75)
        bmSireErg = st.number_input("Sire's Dam ERG", min_value=0, max_value=100, value=75)
        bmSireForm = st.text_input("Sire's Dam Form", value="G2p")

# Predict button
if st.button("🎯 Predict Rating", type="primary", width="stretch"):
    # TODO: Need to preprocess form, damform, sireform, bmsireform

    numericForm = mapFormToHierarchy(pd.Series([form]))[0]
    numericSireForm = mapFormToHierarchy(pd.Series([sireForm]))[0]
    numericDamForm = mapFormToHierarchy(pd.Series([damForm]))[0]
    numericBMSireForm = mapFormToHierarchy(pd.Series([bmSireForm]))[0]

    input_data = pd.DataFrame({
        'form': [numericForm],
        'rawErg': [rawErg],
        'erg': [erg],
        'age': [age],
        'fee': [fee],
        'crop': [crop],
        'ems3': [ems3],
        'damForm': [numericDamForm],
        'sex_C': [1 if sex == 'C' else 0],
        'sex_F': [1 if sex == 'F' else 0],
        'sex_G': [1 if sex == 'G' else 0],
        'sex_R': [1 if sex == 'R' else 0],
        'avgSireRating': [sireRating],
        'avgSireErg': [sireErg],
        'avgSireForm': [numericSireForm],
        'avgDamRating': [damRating],
        'avgDamErg': [damErg],
        'avgBmSireRating': [bmSireRating],
        'avgBmSireErg': [bmSireErg],
        'avgBmSireForm': [numericBMSireForm]
        
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.success(f"## {name} Predicted Rating: {prediction:.2f}")
    
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
st.sidebar.metric("Version", "1.0")