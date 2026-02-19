import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model
model = pickle.load(open('insurance_model.pkl', 'rb'))

st.set_page_config(page_title="Health Insurance Predictor", layout="wide")

st.title("🏥 Health Insurance Claim Predictor")
st.write("Fill in the details below to estimate the insurance claim amount.")

# Creating columns for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)
    blood_pressure = st.number_input("Blood Pressure", 60, 200, 120)

with col2:
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    diabetic = st.selectbox("Diabetic?", ["No", "Yes"])
    smoker = st.selectbox("Smoker?", ["No", "Yes"])
    region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

# Mapping inputs to match the LabelEncoder logic
gender_val = 1 if gender == "Male" else 0
diabetic_val = 1 if diabetic == "Yes" else 0
smoker_val = 1 if smoker == "Yes" else 0
region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
region_val = region_map[region]

# Prediction Button
if st.button("Predict Claim Amount"):
    # Arrange features in the EXACT order: age, gender, bmi, bloodpressure, diabetic, children, smoker, region
    features = np.array([[age, gender_val, bmi, blood_pressure, diabetic_val, children, smoker_val, region_val]])
    
    prediction = model.predict(features)
    
    st.success(f"### Estimated Claim Amount: ${prediction[0]:,.2f}")
    st.info("Note: This is an ML-based estimate and not a final quote.")