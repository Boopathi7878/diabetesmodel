import pandas as pd 
import streamlit as st
import joblib

# Load models
rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')

st.title("Diabetes Prediction Model")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Choose a Model To Predict:",
    ["Random Forest","Support Vector Machine"]
)

st.header("Enter Patient Details")

# User inputs
glucose = st.number_input("Glucose Level:", 0)
bp = st.number_input("Blood Pressure:", 0)
bmi = st.number_input("BMI:", 0.0)
age = st.number_input("Age:", 0)

if st.button("Predict Diabetes:"):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame(
        [[glucose, bp, bmi, age]],
        columns=["Glucose", "Blood Pressure", "BMI", "Age"]
    )
    
    # Model prediction
    if model_choice == "Random Forest":
        result = rf_model.predict(input_data)
    else:
        result = svm_model.predict(input_data)
    
    # Show result
    if result[0] == 1:
        st.error("Sorry, you have Diabetes.")
    else:
        st.success("You don't have Diabetes. Take a sweet!")
