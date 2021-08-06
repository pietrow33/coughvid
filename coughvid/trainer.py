from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import metrics
from google.cloud import storage
import io
import joblib


BUCKET_NAME = 'coughvid-650'
STORAGE_LOCATION = 'models/coughvid/model.joblib'

def get_data():
    client = storage.Client()
    bucket = client.get_bucket('coughvid-650')

    blob = bucket.get_blob('array/data.npy')
    X = np.load(io.BytesIO(blob.download_as_string()))

    blob2 = bucket.get_blob('array/target.npy')
    y = np.load(io.BytesIO(blob2.download_as_string()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

def initialize_model():
    model = models.Sequential()

    #first_convolution
    model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=(288, 432, 4)))
    model.add(layers.MaxPooling2D(2, 2))
    #second_convolution
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    #third_convolution
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    #fourth_convolution
    #     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2,2),
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy',metrics.Recall()])
    
    return model

def train_model(X_train, y_train):
    model = initialize_model()
    model.fit(X_train,y_train, batch_size=32,epochs=50)
    return model

def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

if __name__ == '__main__':
    # get training data from GCP bucket
    X_train, X_test, y_train, y_test = get_data()

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    reg = train_model(X_train, y_train)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(reg)
