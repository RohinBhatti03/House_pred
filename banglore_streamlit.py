import streamlit as st
import pickle
import pandas as pd
import numpy as np


pipe = pickle.load(open('prediction_pickle.pkl', 'rb'))
st.title("Bangalore House Price Predictor")

try:
    data_main = pd.read_csv("bengaluru_house_prices.csv")
except FileNotFoundError:
    st.error("Error: 'bengaluru_house_prices.csv' not found. Please ensure it's in the same directory.")
    st.stop()
    

locations_cleaned = data_main['location'].dropna().astype(str).apply(lambda x: x.strip())


location_counts = locations_cleaned.value_counts()
known_locations = location_counts[location_counts > 10].index.tolist()
known_locations.sort()


location = st.selectbox("Select Location", known_locations, key="location_selector")

total_sqft = st.number_input("Enter Total Square Feet", min_value=500, max_value=10000, step=10)
bath = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Enter Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1)

if st.button("Predict Price"):
    
    final_location = location.strip() 

    
    input_data_df = pd.DataFrame([[final_location, total_sqft, bath, bhk]], 
                             columns=['location', 'total_sqft', 'bath', 'bhk'])

    
    prediction = pipe.predict(input_data_df)[0]
    st.success(f"Estimated Price: ₹ {round(prediction, 2)} Lakhs")