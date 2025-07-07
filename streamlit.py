import pandas as pd
import streamlit as st
import joblib

#load models
rf_model = joblib.load(r'C:\\Users\\Boopathi Kumar\\OneDrive\\Desktop\\Advi-code\\random_forest_model.pkl')
svm_model = joblib.load(r'C:\\Users\\Boopathi Kumar\\OneDrive\\Desktop\\Advi-code\\svm_model.pkl')

#title
st.title("Diabetes Prediction Model")


model_choice = st.sidebar.selectbox("Choose a Model to Predict:", ["Random Forest", "Support Vector Machine"])


st.header("Enter Patient Details")

glucose = st.number_input("Glucose Level:", min_value=0)
bp = st.number_input("BloodPressure:", min_value=0)  # No space in label
bmi = st.number_input("BMI:", min_value=0.0)
age = st.number_input("Age:", min_value=0)

#prediction
if st.button("Predict Diabetes"):
    #create input DataFrame with matching feature names
    input_data = pd.DataFrame([[glucose, bp, bmi, age]],
                              columns=["Glucose", "BloodPressure", "BMI", "Age"])  # Column names must match those used during training

    #select model and predict
    if model_choice == "Random Forest":
        result = rf_model.predict(input_data)
    else:
        result = svm_model.predict(input_data)

    #display result
    if result[0] == 1:
        st.error("Sorry, you have Diabetes.")
    else:
        st.success("You don't have Diabetes. Take a sweet!")
