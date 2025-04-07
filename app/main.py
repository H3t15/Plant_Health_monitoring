import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from supabase import create_client

# ---------------------
# Setup
# ---------------------
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndzenFnZ3phZnRyZmJ2amJ1dWF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM5MTYyMzEsImV4cCI6MjA1OTQ5MjIzMX0.ZAgvlXIN4X28nxCh3-0rKW8v9jqcOI24UMv2_fm6J_8'
API_URL = 'https://wszqggzaftrfbvjbuuat.supabase.co'
supabase = create_client(API_URL, API_KEY)

st.set_page_config(page_title="Plant Health Monitoring", layout='wide')

# ---------------------
# Session
# ---------------------
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None

# ---------------------
# Custom Styles
# ---------------------
st.markdown("""
    <style>
        .stApp {
            background-color: #eafbee;
            color: #0b3d0b;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-header {
            background-color: #2e7d32;
            color: white;
            padding: 1rem;
            text-align: center;
            font-size: 28px;
            border-radius: 5px;
        }
        .stSidebar > div:first-child {
            background-color: #a5d6a7;
        }
        .css-10trblm, .css-1v3fvcr, .css-qrbaxs {
            color: #0b3d0b !important;
        }
        .stTextInput > div > div > input,
        .stTextInput > div > div > textarea,
        .stButton > button {
            background-color: white !important;
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: black !important;
        }
        .stSuccess, .stWarning, .stError {
            background-color: #263238 !important;
            border-color: #4caf50 !important;
            color: #e8f5e9 !important;
        }
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌱 Plant Health Monitoring System</div>', unsafe_allow_html=True)

# ---------------------
# Navigation
# ---------------------
if st.session_state['authenticated']:
    nav_selection = st.sidebar.radio("Navigation", ["Home", "Plant Disease Classifier", "Sensor Dashboard", "Logout"])
else:
    nav_selection = st.sidebar.radio("Navigation", ["Home", "Login", "Register"])

# ---------------------
# Home Page
# ---------------------
if nav_selection == "Home":
    st.title("Welcome to the Plant Health Monitoring Dashboard")
    st.markdown("""
        Stay on top of your garden's health with real-time sensor data and intelligent plant disease prediction powered by AI.
    """)

# ---------------------
# Register Page
# ---------------------
elif nav_selection == "Register":
    st.title("👤 Register")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        try:
            res = supabase.auth.sign_up({"email": email, "password": password})
            if res.user:
                st.success("Registered successfully! Please verify your email.")
            else:
                st.error("Registration failed.")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------
# Login Page
# ---------------------
elif nav_selection == "Login":
    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if res.user:
                st.session_state['authenticated'] = True
                st.session_state['user'] = res.user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Login failed.")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------
# Logout
# ---------------------
elif nav_selection == "Logout":
    supabase.auth.sign_out()
    st.session_state['authenticated'] = False
    st.session_state['user'] = None
    st.success("Logged out successfully.")
    st.rerun()

# ---------------------
# Plant Disease Classifier
# ---------------------
elif nav_selection == "Plant Disease Classifier" and st.session_state['authenticated']:
    st.title("🌿 Plant Disease Classifier")
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
    model = tf.keras.models.load_model(model_path)
    class_indices = json.load(open(f"{working_dir}/trained_model/class_indices.json"))

    def load_and_preprocess_image(image_path, target_size=(224, 224)):
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.
        return img_array

    def predict_image_class(model, image_path, class_indices):
        preprocessed_img = load_and_preprocess_image(image_path)
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        return class_indices[str(predicted_class_index)]

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, width=200)
        if st.button("Classify"):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: {prediction}")

# ---------------------
# Sensor Dashboard
# ---------------------

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

if nav_selection == "Sensor Dashboard" and st.session_state.get('authenticated', False):
    st.title("📊 Sensor Data Dashboard")

    # Step 1: Fetch data from Supabase
    supabase_data = supabase.table('SensorData').select('*').execute().data
    df = pd.DataFrame(supabase_data)

    if df.empty:
        st.warning("⚠️ No sensor data available.")
        st.stop()

    # Step 2: Show raw data
    st.write("📦 Raw Data Preview:", df.head())
    st.write("🔍 Original Columns from Supabase:", df.columns.tolist())

    # Step 3: Fix typos and standardize column names
    df.rename(columns={
        "Temprature": "air_temperature",   # Correct typo
        "Temperature": "air_temperature",  # In case database fixes it
        "STemprature": "soil_temperature"  # Correct typo
    }, inplace=True)

    # Step 4: Normalize all column names
    df.columns = df.columns.str.strip().str.lower()
    st.write("✅ Final Normalized Columns:", df.columns.tolist())

    # Step 5: Convert created_at to datetime
    if "created_at" in df.columns:
        df["datetime"] = pd.to_datetime(df["created_at"])
    else:
        st.error("❌ 'created_at' column is missing!")
        st.stop()

    # Step 6: Define sensor columns for plotting
    sensor_columns = {
        "air_temperature": "🌬️ Air Temperature (DHT22)",
        "soil_temperature": "🌡️ Soil Temperature (OneWire)",
        "humidity": "💧 Humidity (DHT22)",
        "soilmoisture": "🌱 Soil Moisture (%)"
    }

    # Step 7: Plot each sensor data
    for col, label in sensor_columns.items():
        if col in df.columns:
            # Convert to numeric if needed
            df[col] = pd.to_numeric(df[col], errors="coerce")
            plot_df = df.dropna(subset=["datetime", col]).sort_values("datetime")

            # Debugging: show values
            st.write(f"🔎 `{col}` raw values:", df[col].tolist())
            st.write(f"✅ `{col}` filtered data:", plot_df[[col]].head())

            if not plot_df.empty:
                st.markdown(f"### {label}")
                fig, ax = plt.subplots()
                sns.lineplot(data=plot_df, x="datetime", y=col, ax=ax, marker='o', color='tab:blue')
                ax.set_xlabel("Timestamp")
                ax.set_ylabel(label)
                ax.set_title(label)
                fig.autofmt_xdate()
                st.pyplot(fig)
            else:
                st.warning(f"⚠️ No valid data to plot for `{col}`.")
        else:
            st.warning(f"❗ Column `{col}` not found in the data.")
