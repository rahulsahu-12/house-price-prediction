import streamlit as st
import pandas as pd
import joblib

# Load model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

st.title("California House Price Predictor 🏠")

# Create form inputs
longitude = st.number_input("Longitude", -125.0, -113.0, -120.0)
latitude = st.number_input("Latitude", 32.0, 42.0, 37.0)
housing_median_age = st.number_input("Housing Median Age", 1, 100, 20)
total_rooms = st.number_input("Total Rooms", 1, 10000, 2000)
total_bedrooms = st.number_input("Total Bedrooms", 1, 3000, 500)
population = st.number_input("Population", 1, 10000, 1000)
households = st.number_input("Households", 1, 5000, 400)
median_income = st.number_input("Median Income", 0.0, 20.0, 3.5)

ocean_proximity = st.selectbox("Ocean Proximity", [
    "NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"
])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    transformed_input = pipeline.transform(input_df)
    prediction = model.predict(transformed_input)

    st.success(f"🏡 Estimated House Price: ${int(prediction[0]):,}")
