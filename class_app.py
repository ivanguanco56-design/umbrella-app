# ================= IMPORT LIBRARIES =================
import streamlit as st
import numpy as np
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Umbrella Predictor App",
    page_icon="🌦",
    layout="centered"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("umbrella_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ================= TITLE =================
st.title("AI Umbrella Prediction System")
st.write("Enter weather conditions to predict whether you need an umbrella.")

# ================= INPUT SECTION =================
st.subheader("Input Weather Data")

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=25.0
)

pm25 = st.number_input(
    "Air Quality PM2.5",
    min_value=0.0,
    max_value=500.0,
    value=50.0
)

# ================= PREDICTION =================
if st.button("Predict"):

    input_data = np.array([[temperature, pm25]])
    prediction = model.predict(input_data)

    st.subheader("Result")

    if prediction[0] == 1:
        st.error("You should bring an umbrella.")
        st.write("Weather conditions indicate possible rain or discomfort.")
    else:
        st.success("No umbrella needed.")
        st.write("Weather conditions are safe.")

# ================= FOOTER =================
st.markdown("---")
st.caption("AI Weather Decision System | Capstone Project")