import streamlit as st
import pickle
import numpy as np

# Load the Random Forest model
with open('tunedRF.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Prediction App", layout="wide")

st.title("🚀 Random Forest Prediction App")
st.write("Fill out the features in the sidebar and click Predict.")
st.write(f"DEBUG: Model expects {model.n_features_in_} features.")  # Show expected number

# Sidebar for Inputs
st.sidebar.header("Enter Feature Values")

input_data = []
for i in range(model.n_features_in_):  # 💥 Loop based on model's needs
    value = st.sidebar.slider(f'Feature {i+1}', min_value=0.0, max_value=100.0, value=0.0)
    input_data.append(value)

# Main Prediction Area
st.subheader("Prediction Result")

if st.button('Predict'):
    prediction = model.predict(np.array([input_data]))
    st.success(f"🎯 Predicted Value: {prediction[0]}")


