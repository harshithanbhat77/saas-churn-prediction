import streamlit as st
import pandas as pd
from model_inference import load_model, predict_new
from data_preprocessing import preprocess_data

st.title("SaaS Churn Prediction App")

st.markdown("""
Enter customer details below to predict the likelihood of churn. The app will show the prediction and the model's confidence.
""")

# Collect user input for all features except customerID and Churn
gender = st.selectbox('Gender', ['Female', 'Male'])
senior_citizen = st.selectbox('Senior Citizen', [0, 1])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=1)
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=0.0)
total_charges = st.number_input('Total Charges', min_value=0.0, value=0.0)

# Prepare input as DataFrame
input_dict = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_dict])

st.subheader("Input Summary")
st.write(input_df)

if st.button('Predict Churn'):
    try:
        # Preprocess input
        df_processed = preprocess_data(input_df)
        # Load model
        model = load_model()
        # Predict
        prediction = predict_new(model, df_processed)
        # Probability/confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_processed)[0][1]
            st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'Not Churn'} (Confidence: {proba:.2%})")
        else:
            st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'Not Churn'}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
