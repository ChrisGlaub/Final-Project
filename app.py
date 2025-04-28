import streamlit as st
import pickle

# Load the Random Forest model
with open('tunedRF.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Random Forest Prediction App")
st.write(f"Model expects {model.n_features_in_} features.")

# Example input fields
st.subheader("Input Features")
feature1 = st.number_input('Feature 1', value=0.0)
feature2 = st.number_input('Feature 2', value=0.0)
feature3 = st.number_input('Feature 3', value=0.0)
feature4 = st.number_input('Feature 4', value=0.0)

# Prediction
if st.button('Predict'):
    input_data = [[feature1, feature2, feature3, feature4]]  # must match model input size
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
