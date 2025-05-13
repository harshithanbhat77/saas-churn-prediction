import streamlit as st
import joblib
import numpy as np

model = joblib.load('../models/churn_model.pkl')

st.title("SaaS Churn Prediction")

login_freq = st.slider('Login Frequency (per week)', 0, 20, 3)
plan_type = st.selectbox('Plan Type', ['Basic', 'Pro', 'Enterprise'])

# You would encode this properly
features = np.array([[login_freq, 1]])  # Dummy input for now

if st.button('Predict'):
    prediction = model.predict(features)
    st.write("Churn" if prediction[0] == 1 else "Not Churn")
