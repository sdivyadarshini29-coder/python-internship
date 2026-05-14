import streamlit as st
import pickle
import numpy as np

# 1. Train panna model-ah load panrom
model = pickle.load(open('iris_svm_model.pkl', 'rb'))

# Species names mapping
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

st.title("🌸 Iris Flower Classification")
st.write("Enter the measurements to find the species:")

# 2. Input Sliders (User input eduka)
sepal_l = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_w = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_l = st.slider('Petal Length (cm)', 1.0, 7.0, 1.5)
petal_w = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

# 3. Predict Button logic
if st.button("Predict Species"):
    features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    prediction = model.predict(features)
    result = species_map[prediction[0]]
    
    st.success(f"The Predicted Species is: **{result}**")