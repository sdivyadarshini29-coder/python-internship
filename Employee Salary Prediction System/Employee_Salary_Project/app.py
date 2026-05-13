import streamlit as st
import pandas as pd
import pickle

# Load the saved model
model = pickle.load(open('salary_model.pkl', 'rb'))

st.title("👨‍💼 Employee Salary Predictor")
st.write("Enter the details below to estimate the salary:")

# User Inputs
exp = st.number_input("Years of Experience", min_value=0, max_value=40, value=1)
edu = st.selectbox("Education Level", [1, 2, 3], format_func=lambda x: "Bachelor" if x==1 else "Master" if x==2 else "PhD")
age = st.slider("Age", 18, 65, 25)

if st.button("Predict Salary"):
    input_data = pd.DataFrame([[exp, edu, age]], columns=['Years_Experience', 'Education_Level', 'Age'])
    prediction = model.predict(input_data)
    st.success(f"Estimated Salary: ₹{round(prediction[0], 2)}")