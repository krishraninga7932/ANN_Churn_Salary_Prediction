import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# Page Config
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)


# Load the trained model
model = tf.keras.models.load_model('model.h5')


# Load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Title
st.title("Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to leave the bank.")
st.divider()


# Input Section
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

with col2:
    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=70000.0,
        step=1000.0
    )

    num_of_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

with col3:
    geography = st.selectbox(
        "Geography",
        onehot_encoder_geo.categories_[0]
    )

    gender = st.selectbox(
        "Gender",
        label_encoder_gender.classes_
    )

    has_cr_card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    is_active_member = st.selectbox(
        "Is Active Member",
        [0, 1]
    )


st.divider()


# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# Geography encoding
geo_encoder = onehot_encoder_geo.transform([[geography]])

geo_encoded_df = pd.DataFrame(
    geo_encoder,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)


# Combine OHE columns
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)


# Scale
input_scaled = scaler.transform(input_data)


# Prediction
if st.button("Predict Churn", use_container_width=True):

    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    st.subheader("Prediction Result")

    if prediction_proba > 0.5:
        st.error(
            f"Customer is likely to churn ({prediction_proba:.2%})"
        )
    else:
        st.success(
            f"Customer is not likely to churn ({prediction_proba:.2%})"
        )

    st.progress(float(prediction_proba))

    st.metric(
        label="Churn Probability",
        value=f"{prediction_proba:.2%}"
    )