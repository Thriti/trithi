import streamlit as st
import numpy as np
import joblib

# Load the model
loaded_model = joblib.load('Diabetes_Prediction_Model.pkl')

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    sample_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = loaded_model.predict(sample_data)
    return 'Diabetes' if prediction[0] == 1 else 'No Diabetes'

# Streamlit App Interface
st.title("Diabetes Prediction - Trithi ")

# Input parameters on the main page (not in the sidebar)
Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=6)
Glucose = st.number_input('Glucose', min_value=0, max_value=200, value=148)
BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=130, value=72)
SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=35)
Insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0)
BMI = st.number_input('BMI', min_value=0.0, max_value=100.0, value=33.6)
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.627)
Age = st.number_input('Age', min_value=1, max_value=120, value=50)

# Button for prediction
if st.button('Predict'):
    result = predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    st.success(f"The model predicts: {result}")
