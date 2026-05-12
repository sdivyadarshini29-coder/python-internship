import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to see if they would have survived.")

# Create input fields
pclass = st.selectbox("Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)

# Convert sex to numerical
sex_val = 1 if sex == "female" else 0

if st.button("Predict"):
    # Arrange features for the model
    features = np.array([[pclass, sex_val, age, sibsp, parch]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("The passenger likely survived! 🎉")
    else:
        st.error("The passenger likely did not survive. 😔")