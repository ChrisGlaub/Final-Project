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
    file_id = "1ij4RLvCK9KlCmwNA5qDXvl6S1bMBWb1z"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# —————————————
# 1. Load your model
# —————————————
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# … rest of your app …





