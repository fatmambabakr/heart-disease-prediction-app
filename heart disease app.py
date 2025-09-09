import streamlit as st
import joblib
import pandas as pd

# ---------- Load the trained pipeline ----------
model = joblib.load("heart_disease_modelv2 (1).pkl")  

# ---------- Streamlit Title ----------
st.title("💓 Heart Disease Prediction App")
st.write("Please enter your medical data to generate a prediction")

# ---------- User Inputs ----------
age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox("Gender", ["Female", "Male"])
sex_input = 0 if sex == "Female" else 1
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholestoral", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels colored by fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [3, 6, 7])  # خليتيه زي الموديل

# ---------- Prepare features in the same order as trained pipeline ----------
features = pd.DataFrame([[
    age, trestbps, chol, thalach, oldpeak,
    1 if ca == 1 else 0,
    1 if ca == 2 else 0,
    1 if ca == 3 else 0,
    1 if thal == 6 else 0,
    1 if thal == 7 else 0
]], columns=[
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak',
    'ca_1.0', 'ca_2.0', 'ca_3.0', 'thal_6.0', 'thal_7.0'
])

# ---------- Prediction ----------
if st.button("Predict"):
    pred = model.predict(features)[0]
   

    if pred == 0:
        st.success("No Heart Disease ❤️")
    elif pred == 1:
        st.info("Mild Heart Disease ⚠️")
    elif pred == 2:
        st.warning("Moderate Heart Disease ⚠️")
    elif pred == 3:
        st.error("Severe Heart Disease 🚨")
    elif pred == 4:
        st.error("Very Severe Heart Disease 🚨🔥")
    else:
        st.write("Unknown Class")


