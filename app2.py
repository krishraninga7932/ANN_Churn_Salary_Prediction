import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model("regression_model.h5")

# Load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Page Config
st.set_page_config(
    page_title="Salary Regression Prediction",
    layout="centered"
)

st.title("Salary Regression Prediction")

# Input Fields
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox(
        "Geography",
        onehot_encoder_geo.categories_[0]
    )

    gender = st.selectbox(
        "Gender",
        label_encoder_gender.classes_
    )

    age = st.slider(
        "Age",
        18,
        100,
        35
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=1000,
        value=650
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

with col2:
    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

    num_of_products = st.slider(
        "Number Of Products",
        1,
        4,
        2
    )

    has_cr_card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    is_active_member = st.selectbox(
        "Is Active Member",
        [0, 1]
    )

    exited = st.selectbox(
        "Exited",
        [0, 1]
    )

# Create DataFrame
input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Exited": [exited]
})

# Geography Encoding
geo_encoded = onehot_encoder_geo.transform([[geography]])

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(
        ["Geography"]
    )
)

# Combine Data
input_data = pd.concat(
    [input_df.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Scale
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Salary"):

    prediction = model.predict(input_scaled)

    predicted_salary = prediction[0][0]

    st.success(
        f"Predicted Salary: ₹ {predicted_salary:,.2f}"
    )