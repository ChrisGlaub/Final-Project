import streamlit as st
import pickle
import numpy as np
import locale

# Load the trained Random Forest model
with open('final_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up the page and title
st.set_page_config(page_title="Prediction App", layout="wide")
st.title("Random Forest Prediction App")
st.write("Fill out the features in the sidebar and click Predict.")

# Sidebar for Inputs
st.sidebar.header("Enter Feature Values")
feature1 = st.sidebar.slider('Feature 1', min_value=0.0, max_value=100.0, value=0.0)
feature2 = st.sidebar.slider('Feature 2', min_value=0.0, max_value=100.0, value=0.0)
feature3 = st.sidebar.slider('Feature 3', min_value=0.0, max_value=100.0, value=0.0)

# Include other country features as binary
# For example:
# feature4 = st.sidebar.selectbox('Country: United States', options=[0, 1])

# Combine the inputs into an array
input_data = [feature1, feature2, feature3]  # Add other features as needed
input_data = np.array([input_data])  # Convert to numpy array for model prediction

# Main Prediction Area
st.subheader("Prediction Result")

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_data)

    # Format the output as a salary
    locale.setlocale(locale.LC_ALL, '')
    formatted_salary = locale.currency(prediction[0], grouping=True)
    st.success(f"Predicted Salary: {formatted_salary}")




