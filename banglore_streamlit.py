import streamlit as st
import pickle
import pandas as pd
import numpy as np

pipe = pickle.load(open('prediction_pickle.pkl', 'rb'))
st.title("Bangalore House Price Predictor")

data_main = pd.read_csv("bengaluru_house_prices.csv")
locations = sorted(data_main['location'].dropna().astype(str).unique())
location = st.selectbox("Select Location", locations)
total_sqft = st.number_input("Enter Total Square Feet", min_value=500, max_value=10000, step=10)
bath = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Enter Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1)

if st.button("Predict Price"):
# Create a single-row DataFrame with the exact column names
    input_data_df = pd.DataFrame([[location, total_sqft, bath, bhk]], 
                             columns=['location', 'total_sqft', 'bath', 'bhk'])

# Pass the DataFrame to the predict method
    prediction = pipe.predict(input_data_df)[0]
    st.success(f"Estimated Price: ₹ {round(prediction, 2)} Lakhs")
