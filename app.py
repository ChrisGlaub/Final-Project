import streamlit as st
import pickle
import numpy as np
import locale

# —————————————
# 1. Load your model
# —————————————
with open('final_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# —————————————
# 2. Page configuration
# —————————————
st.set_page_config(page_title="Salary Prediction App", layout="wide")
st.title("💼 Salary Prediction App")
st.write("Use the sidebar to enter your info, then click Predict")

# —————————————
# 3. Sidebar inputs
# —————————————
st.sidebar.header("Enter Your Details")

# 3a) Numeric features: (label, min, max, step)
numeric_features = [
    ("Years of Coding Experience", 0.0, 50.0, 1.0),
    ("Years of Machine Learning Experience", 0.0, 50.0, 1.0),
    ("Money Spent on ML/Cloud in Last 5 Years ($)", 0, 100_000, 1_000),
    # … add any other continuous inputs here …
]

inputs = []
for label, vmin, vmax, step in numeric_features:
    val = st.sidebar.slider(label, min_value=vmin, max_value=vmax, value=vmin, step=step)
    inputs.append(val)

# 3b) Country (one‐hot):
countries = [
    "United States of America","Canada","United Kingdom","France",
    "Germany","India","Brazil","China","Japan","Australia",
    # … fill in all ~20 country names exactly as your dummy columns …
]
country = st.sidebar.selectbox("Country", countries)
for c in countries:
    inputs.append(1 if c == country else 0)

# —————————————
# 4. Predict button
# —————————————
if st.sidebar.button("Predict"):
    # Convert to 2D array for sklearn
    X_new = np.array([inputs])
    pred = model.predict(X_new)[0]

    # Format as currency
    locale.setlocale(locale.LC_ALL, '')            # use system locale
    salary = locale.currency(pred, grouping=True)

    # Display result
    st.subheader("🎯 Predicted Salary")
    st.success(salary)





