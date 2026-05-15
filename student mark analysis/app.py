import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Page Configuration ---
st.set_page_config(page_title="EduSmart AI", layout="wide", page_icon="🎓")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .prediction-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- Data & Model Loading ---
@st.cache_resource
def load_resources():
    # Ensure these files exist in your folder after running model_train.py
    model = joblib.load('score_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    df = pd.read_csv('online_vs_offline_learning_dataset.csv')
    return model, scaler, features, df

try:
    model, scaler, features, df = load_resources()
except Exception as e:
    st.error("Error: Model files not found. Please run 'python model_train.py' first.")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=100)
st.sidebar.title("Navigation Menu")
page = st.sidebar.radio("Select Page", ["🏠 Home", "ℹ️ About Project", "🔮 Prediction", "📊 Data Analysis"])

# ----------------- 1. HOME PAGE -----------------
if page == "🏠 Home":
    st.title("🎓 Student Performance Analytics")
    st.markdown("### Predict Exam Scores based on Learning Habits")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Welcome to **EduSmart AI**! This application analyzes student study behavior such as 
        Study Hours, Focus Levels, and Retention Scores to predict potential Exam Results.
        
        It also compares the effectiveness of **Online vs Offline** learning modes using 
        advanced Machine Learning algorithms.
        
        **How to use this app:**
        1. Select the **Prediction** page from the sidebar.
        2. Input your daily study metrics.
        3. Click 'Calculate' to view your predicted score and personalized suggestions.
        """)
        if st.button("Get Started"):
            st.info("Please select the 'Prediction' tab from the sidebar menu to begin.")
    with col2:
        st.image("https://img.freepik.com/free-vector/students-watching-webinar-computer-studying-online_74855-15522.jpg")

# ----------------- 2. ABOUT PAGE -----------------
elif page == "ℹ️ About Project":
    st.title("Project Overview")
    st.info("Goal: To provide data-driven insights into how different learning environments affect academic performance.")
    
    tab1, tab2 = st.tabs(["Dataset Details", "Technology Stack"])
    with tab1:
        st.write("This project utilizes the **Online vs Offline Learning Dataset** sourced from Kaggle.")
        st.write("### Data Preview:")
        st.dataframe(df.head(10))
    with tab2:
        st.write("### Technologies Used:")
        st.write("- **Language:** Python")
        st.write("- **Library:** Scikit-Learn (Machine Learning)")
        st.write("- **Algorithms:** Linear Regression, Logistic Regression, SVM, KNN")
        st.write("- **Web Framework:** Streamlit")

# ----------------- 3. PREDICTION PAGE -----------------
elif page == "🔮 Prediction":
    st.title("Exam Score Prediction Engine")
    st.write("Please enter your study details below to estimate your performance:")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            mode = st.selectbox("Learning Mode", ["Online", "Offline"])
            subject = st.selectbox("Primary Subject", ["Math", "English", "Science", "History", "Programming"])
            hours = st.slider("Daily Study Hours", 1, 12, 5)
        with col2:
            focus = st.slider("Focus Level (0-100%)", 0, 100, 75)
            retention = st.slider("Retention Level (0-100%)", 0, 100, 60)
            
    if st.button("Predict My Score"):
        with st.spinner('AI Engine is processing your data...'):
            time.sleep(1.5)
            
            # Encoding logic
            mode_val = 1 if mode == "Online" else 0
            # Pre-defined feature array (Matches the training features)
            input_array = [mode_val, hours, retention, focus, 0, 0, 0, 0] 
            scaled_data = scaler.transform([input_array])
            prediction = model.predict(scaled_data)[0]
            
            # Constraints to keep score within 0-100 range
            final_score = max(0, min(100, prediction))
            
            st.balloons()
            st.success(f"### 🎯 Your Predicted Exam Score: {final_score:.2f}")

            # --- EXTRA FEATURE: SMART SUGGESTIONS ---
            st.subheader("💡 Smart Recommendations")
            if hours < 4:
                st.warning("Your study duration is quite low. We recommend dedicating at least 4-5 hours daily for better results.")
            elif focus > 85 and final_score > 80:
                st.success("Excellent! Your focus and predicted score are outstanding. Maintain this consistency!")
            else:
                st.info("To improve focus and retention, consider using active recall techniques or meditation.")

# ----------------- 4. RESULTS / ANALYSIS PAGE -----------------
elif page == "📊 Data Analysis":
    st.title("Visual Data Insights")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Samples Analyzed", len(df))
    m2.metric("Average Exam Score", f"{round(df['Exam_Score'].mean(), 2)}%")
    m3.metric("Average Focus Level", f"{round(df['Focus_Level'].mean(), 1)}%")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### Learning Mode vs. Performance")
        fig, ax = plt.subplots()
        sns.boxplot(x='Learning_Mode', y='Exam_Score', data=df, palette="viridis")
        st.pyplot(fig)
        st.caption("This chart shows the distribution of scores between Online and Offline students.")
        
    with c2:
        st.write("#### Study Hours vs. Information Retention")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Study_Hours', y='Retention_Score', hue='Learning_Mode', data=df)
        st.pyplot(fig)
        st.caption("Analyzing the correlation between time spent studying and memory retention.")