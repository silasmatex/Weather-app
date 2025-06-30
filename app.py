import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/weather_model.pkl")

# Mapping encoded target to labels
condition_map = {0: 'Cloudy', 1: 'Rainy', 2: 'Stormy', 3: 'Sunny'}

st.title("🌦️ Cameroon Weather Prediction App")
st.write("Predict the weather condition based on temperature, humidity, rainfall, and wind speed.")

# Input fields
temp = st.slider("Temperature (°C)", 20, 40, 30)
humidity = st.slider("Humidity (%)", 10, 100, 70)
rainfall = st.slider("Rainfall (mm)", 0.0, 100.0, 10.0)
wind_speed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0)

if st.button("Predict"):
    input_data = pd.DataFrame([[temp, humidity, rainfall, wind_speed]], 
                              columns=['Temp', 'Humidity', 'Rainfall', 'WindSpeed'])
    prediction = model.predict(input_data)
    condition = condition_map[prediction[0]]
    st.success(f"Predicted Weather: {condition}")
