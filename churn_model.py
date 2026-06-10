import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime



#load the dataset
data=pd.read_csv("Churn_Modelling.csv")

# drop irrelevant feaures
data=data.drop(['RowNumber','CustomerId','Surname'],axis=1)
# print(data.head())

# encode categorical variable needs 1d array so []
label_encoder_gender=LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])
# print(data)

# one hot encodeing geography needs 2d array so [[]]
onehot_encoder_geo=OneHotEncoder(sparse_output=False)
geo_encoder=onehot_encoder_geo.fit_transform(data[['Geography']])
geo_encoded_df = pd.DataFrame(
    geo_encoder,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)
# print(geo_encoded_df)


# combine all the columns with the original data
data = pd.concat([data.drop("Geography",axis=1),geo_encoded_df],axis=1)
# print(data.head())


# save the encoders and scaler
with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)

with open('onehot_encoder_geo.pkl','wb') as file:
    pickle.dump(onehot_encoder_geo,file)
    
    
    
# divide the dataset into independent and dependent features
x=data.drop('Exited',axis=1)
y=data['Exited']


# split the data in training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# scale these features
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# print(x_train)
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)



# ann tensorflow
model=Sequential([
    Dense(64,activation="relu",input_shape=(x_train.shape[1],)), #HL1 connected with input layer
    Dense(32,activation="relu"),#HL2 
    Dense(1,activation="sigmoid"), #output layer
])

# print(model.summary())




# compile the modal

opt=tf.keras.optimizers.Adam(learning_rate=0.01)
loss=tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])



# setup the tensorboard

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)



# setup early stopping
early_stopping_callback=EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)



# training the model
history=model.fit(
    x_train,y_train,validation_data=(x_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)


# save
model.save('model.h5')



 