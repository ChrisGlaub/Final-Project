import streamlit as st
import pickle
import numpy as np

# Load the Random Forest model
with open('tunedRF.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Prediction App", layout="wide")

st.title("🚀 Random Forest Prediction App")
st.write("Fill out the features in the sidebar and click Predict.")

# Sidebar for Inputs
st.sidebar.header("Enter Feature Values")

feature1 = st.sidebar.slider('Feature 1', min_value=0.0, max_value=100.0, value=0.0)
feature2 = st.sidebar.slider('Feature 2', min_value=0.0, max_value=100.0, value=0.0)
feature3 = st.sidebar.slider('Feature 3', min_value=0.0, max_value=100.0, value=0.0)
feature4 = st.sidebar.slider('Feature 4', min_value=0.0, max_value=100.0, value=0.0)

input_data = [feature1, feature2, feature3, feature4]

# Main Prediction Area
st.subheader("Prediction Result")

# Fix: everything inside the button
if st.button('Predict'):
    prediction = model.predict(np.array([input_data]))
    st.success(f"🎯 Predicted Value: {prediction[0]}")

