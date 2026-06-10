# Customer Churn Prediction using ANN

## Overview

This project is a Customer Churn Prediction web application built using Artificial Neural Networks (ANN), TensorFlow, and Streamlit.

The application predicts whether a bank customer is likely to leave the bank based on customer information such as credit score, age, balance, geography, tenure, and other banking-related attributes.

The model is trained on the Churn Modelling dataset and deployed through an interactive Streamlit interface.

---

## Features

* Predict customer churn probability
* Interactive Streamlit user interface
* ANN model built using TensorFlow/Keras
* Data preprocessing using Scikit-Learn
* One-Hot Encoding for Geography
* Label Encoding for Gender
* Feature Scaling using StandardScaler
* Real-time predictions
* Churn probability visualization

---

## Tech Stack

### Frontend

* Streamlit

### Machine Learning

* TensorFlow / Keras
* Scikit-Learn

### Data Processing

* Pandas
* NumPy

---

## Project Structure

```text
├── app.py
├── churn_model.py
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The model is trained on the Churn Modelling dataset containing customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Active Member Status
* Estimated Salary
* Churn Status (Target Variable)

Target Variable:

```text
Exited
0 = Customer Stayed
1 = Customer Left
```

---

## Machine Learning Workflow

### Data Preprocessing

* Removed unnecessary columns
* Encoded Gender using LabelEncoder
* Encoded Geography using OneHotEncoder
* Scaled numerical features using StandardScaler

### Model Architecture

Input Layer

```text
Dense(64, ReLU)
```

Hidden Layer

```text
Dense(32, ReLU)
```

Output Layer

```text
Dense(1, Sigmoid)
```

### Optimizer

```text
Adam
```

### Loss Function

```text
Binary Crossentropy
```

### Evaluation Metric

```text
Accuracy
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd customer-churn-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will start locally and open in your browser.

---

## How It Works

1. User enters customer information.
2. Input data is encoded and scaled.
3. The trained ANN model processes the data.
4. Churn probability is generated.
5. The application displays whether the customer is likely to churn.

---

## Learning Objectives

This project demonstrates:

* Data preprocessing
* Feature engineering
* Label Encoding
* One-Hot Encoding
* Feature Scaling
* Artificial Neural Networks (ANN)
* Binary Classification
* Model Deployment using Streamlit

---

## Future Improvements

* Improved UI/UX
* Model comparison with Random Forest and XGBoost
* Database integration
* User authentication
* Cloud deployment
* Explainable AI visualizations

##

Built as part of a Deep Learning and Generative AI learning journey using TensorFlow and Streamlit.
