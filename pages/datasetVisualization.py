import streamlit as st
import pandas as pd
import numpy as np

st.title("Training dataset")

@st.cache_data
def load_data():
    return pd.read_csv('data/encodedHorseData.csv')

df = load_data()

st.dataframe(df, width=1000, height=600)