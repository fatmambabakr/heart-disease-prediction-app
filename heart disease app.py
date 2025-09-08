import streamlit as st
import joblib
import numpy as np

# ---------- Load the trained pipeline ----------
# استخدمي نفس الاسم اللي حفظتيه
model = joblib.load("heart_disease_modelv2.pkl")  # أو 'heart_disease_model_v2.pkl'

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
thal = st.selectbox("Thalassemia", [1, 2, 3])

# ---------- Prepare features in the same order as trained pipeline ----------
features = np.array([[age, sex_input, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak, slope, ca, thal]])

# ---------- Prediction ----------
if st.button("Predict"):
    pred = model.predict(features)
    prediction = pred[0]

    # Probabilities لكل class
    proba = model.predict_proba(features)
    st.write("Class probabilities:", proba)

    # الرسالة للمستخدم
    if prediction == 0:
        st.success("No Heart Disease ❤️")
    elif prediction == 1:
        st.info("Mild Heart Disease ⚠️")
    elif prediction == 2:
        st.warning("Moderate Heart Disease ⚠️")
    elif prediction == 3:
        st.error("Severe Heart Disease 🚨")
    elif prediction == 4:
        st.error("Very Severe Heart Disease 🚨🔥")
    else:
        st.write("Unknown Class")




