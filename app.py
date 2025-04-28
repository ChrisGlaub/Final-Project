import os
import streamlit as st
import pickle
import numpy as np
import locale

# —————————————
# 0. Download model if missing
# —————————————
MODEL_PATH = "final_rf_model.pkl"
if not os.path.exists(MODEL_PATH):
    import gdown
    file_id = "1ij4RLvCK9KlCmwNA5qDXvl6S1bMBWb1z"  # your Drive file ID
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# —————————————
# 1. Load your model
# —————————————
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# —————————————
# 2. Page configuration
# —————————————
st.set_page_config(page_title="Salary Prediction App", layout="wide")
st.title("💼 Salary Prediction App")
st.write("Use the sidebar to enter your details and click Predict.")

# —————————————
# 3. Sidebar inputs
# —————————————
st.sidebar.header("Enter Your Details")

# 3a) Numeric features (label, min, max, step)
numeric_features = [
    ("Years of Coding Experience", 0.0, 50.0, 1.0),
    ("Years of Machine Learning Experience", 0.0, 50.0, 1.0),
    ("Money Spent on ML/Cloud in Last 5 Years ($)", 0, 100_000, 1_000),
]

inputs = []
for label, vmin, vmax, step in numeric_features:
    val = st.sidebar.slider(label, min_value=vmin, max_value=vmax, value=vmin, step=step)
    inputs.append(val)

# 3b) Country one-hot
countries = [
    "United States of America",
    "Canada",
    "United Kingdom of Great Britain and Northern Ireland",
    "France",
    "Germany",
    "India",
    "Brazil",
    "China",
    "Japan",
    "Australia",
    "Mexico",
    "Russia",
    "South Korea",
    "Turkey",
    "Indonesia",
    "Pakistan",
    "Bangladesh",
    "Egypt",
    "Colombia",
    "Spain",
    "Other"
]
country = st.sidebar.selectbox("Country", countries)
for c in countries:
    inputs.append(1 if c == country else 0)

# —————————————
# 4. Prediction
# —————————————
st.subheader("Prediction Result")
if st.sidebar.button("Predict"):
    X_new = np.array([inputs])
    pred = model.predict(X_new)[0]

    locale.setlocale(locale.LC_ALL, '')  
    salary = locale.currency(pred, grouping=True)

    st.success(f"🎯 Predicted Salary: {salary}")






