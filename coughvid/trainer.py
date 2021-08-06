from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import metrics
from google.cloud import storage
import io


BUCKET_NAME = 'coughvid-vteste'
STORAGE_LOCATION = 'models/coughvid/model.h5'
MATRIX_LOCATION = 'models/coughvid/confusion_matrix.png'

def get_data():
    client = storage.Client()
    bucket = client.get_bucket('coughvid-vteste')

    blob = bucket.get_blob('array2/data2.npy')
    X = np.load(io.BytesIO(blob.download_as_string()))

    blob2 = bucket.get_blob('array2/target2.npy')
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
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy',metrics.Recall()])
    
    return model

def train_model(X_train, y_train):
    model = initialize_model()
    model.fit(X_train,y_train, batch_size=32,epochs=50)
    return model

def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.h5')


def save_model(model):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    model.save("model.h5")
    print("Saved model to disk")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

def upload_matrix_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(MATRIX_LOCATION)

    blob.upload_from_filename('confusion_matrix.png')

def evaluate_model(model, X_test, y_test):
    accuracy = model.evaluate(X_test)
    print('n', 'Test_Accuracy:-', accuracy[1])
    pred = model.predict(X_test)
    y_pred = np.round(pred, 0)
    y_true = y_test
    print('confusion matrix')
    print(confusion_matrix(y_true, y_pred))
        #confusion matrix
    f, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt=".0f", ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.savefig("confusion_matrix.png")
    upload_matrix_to_gcp()
    

if __name__ == '__main__':
    # get training data from GCP bucket
    X_train, X_test, y_train, y_test = get_data()

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    model = train_model(X_train, y_train)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(model)
    evaluate_model(model,X_test,y_test)
