import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("heart_disease_model.pkl") 

st.title("💓 Heart Disease Prediction App")

st.write("Please enter your medical data to generate a prediction")


age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox(
    "Gender",
    options=["Female", "Male"]
)


if sex == "Female":
    sex_input = 0
else:
    sex_input = 1

trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholestoral", 100, 600, 200)
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 10.0, 1.0)


features = np.array([[age, sex_input, trestbps, chol, thalach, exang, oldpeak]])


if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("🚨 High risk of heart disease!")
    else:
        st.success("✅ No significant risk of heart disease")


