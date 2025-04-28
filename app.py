import streamlit as st
import pickle
import numpy as np

# Load the Random Forest model
with open('tunedRF.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Salary Prediction App", layout="wide")

st.title("💼 Salary Prediction App")
st.write("Enter your experience and spending details below to estimate your salary!")

st.sidebar.header("Your Information")

# Correct feature labels (3 features)
feature_labels = [
    "Years of Coding Experience",
    "Years of Machine Learning Experience",
    "Money Spent on ML/Cloud in Last 5 Years ($USD)"
]

# Reasonable min/max
feature_min = [0, 0, 0]
feature_max = [50, 50, 100000]

input_data = []
for i in range(model.n_features_in_):
    label = feature_labels[i] if i < len(feature_labels) else f"Feature {i+1}"
    min_val = feature_min[i] if i < len(feature_min) else 0
    max_val = feature_max[i] if i < len(feature_max) else 100
    value = st.sidebar.slider(label, min_value=min_val, max_value=max_val, value=min_val)
    input_data.append(value)

# Main Prediction Area
st.subheader("Prediction Result")

if st.button('Predict'):
    prediction = model.predict(np.array([input_data]))[0]
    salary = f"${prediction:,.2f}"  # Nicely formatted output
    st.success(f"🎯 Predicted Salary: {salary}")



