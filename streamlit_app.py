import streamlit as st
import pickle
import numpy as np

# Load the trained model
try:
    with open("churn_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set up the Streamlit app
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict whether they are likely to churn.")

# Input fields
tenure = st.slider("Tenure (in months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)
contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])

# One-hot encode the contract type
contract_dict = {
    'Month-to-month': [1, 0],
    'One year': [0, 1],
    'Two year': [0, 0]
}
contract_features = contract_dict[contract]

# Prepare input for prediction
try:
    features = np.array([[tenure, monthly_charges, total_charges] + contract_features])
except Exception as e:
    st.error(f"Feature preparation error: {e}")
    st.stop()

# Predict when button is clicked
if st.button("🔮 Predict Churn", key="predict_button"):
    try:
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.warning("⚠️ Prediction: The customer is likely to churn.")
        else:
            st.success("✅ Prediction: The customer is not likely to churn.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
