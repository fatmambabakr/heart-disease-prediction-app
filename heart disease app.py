import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

st.title("💓 Heart Disease Prediction App")
st.write("Please enter your medical data to generate a prediction")

# --- User Inputs ---
age = st.number_input("Age", 18, 100, 30)

sex = st.selectbox("Gender", ["Female", "Male"])
sex_input = 0 if sex == "Female" else 1

cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])  # 0 = typical angina, 3 = asymptomatic
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholestoral", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels colored by fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [1, 2, 3])  # 1 = normal, 2 = fixed defect, 3 = reversible defect

# --- Prepare features ---
features = np.array([[age, sex_input, cp, trestbps, chol, fbs, thalach, exang, oldpeak, slope, ca, thal]])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("🚨 High risk of heart disease!")
    else:
        st.success("✅ No significant risk of heart disease")






