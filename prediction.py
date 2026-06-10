import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np





# load the ann trained model,scaler pickle,ohe
model=load_model('model.h5')


# load the encoder and scaler
with open ('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
    
with open ('scaler.pkl','rb') as file:
    scaler=pickle.load(file)



# eg input data
input_data = {
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 70000
}


# geography encoding
geo_encoder=onehot_encoder_geo.transform([[input_data['Geography']]])
geo_encoded_df = pd.DataFrame(
    geo_encoder,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)



# convert it into df
input_df=pd.DataFrame([input_data])
# print(input_df)



# encode categorical variable
input_df['Gender']=label_encoder_gender.transform(input_df['Gender'])
# print(input_df)




# combine columns
input_data = pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)
# print(input_data)



# scaling the input data
input_scaled=scaler.transform(input_data)
# print(input_scaled)


# prediction
prediction=model.predict(input_scaled)
# print(prediction)



# prediction probability
prediction_proba=prediction[0][0]
# print(prediction_proba)


if prediction_proba > 0.5:
    print("The customer is likely to churn.")
else:
    print("The customer is not likely to churn.")



