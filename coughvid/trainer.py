from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import layers, callbacks, metrics, models, optimizers
from google.cloud import storage
import io


BUCKET_NAME = 'coughvid-650'
STORAGE_LOCATION = 'models/coughvid/model_500.h5'
x_LOCATION = 'models/coughvid/X_test_500.npy'
y_LOCATION = 'models/coughvid/y_test_500.npy'

def get_data():
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    blob = bucket.get_blob('array_reduced/target.npz')
    y = np.load(io.BytesIO(blob.download_as_string()))
    y = y['arr_0']

    blob = bucket.get_blob('array_reduced/data.npz')
    X = np.load(io.BytesIO(blob.download_as_string()))
    X = X['arr_0']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train = X_train / 255.
    X_test = X_test / 255.
    return X_train, X_test, y_train, y_test

def initialize_model():
    model = models.Sequential()

    #first_convolution
    model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=(217, 334, 4)))
    model.add(layers.MaxPooling2D(2, 2))
    #second_convolution
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    #third_convolution
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    #fourth_convolution
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    opt = optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999)
    model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy', metrics.Recall()])
    
    return model

def train_model(X_train, y_train):
    model = initialize_model()
    value, counts = np.unique(y_train, return_counts=True)
    weight_COVID = counts[0] / counts[1]
    weights = {value[0]: 1, value[1]: weight_COVID}
    ers = callbacks.EarlyStopping(monitor ="val_recall", patience = 50, restore_best_weights = True)
    model.fit(X_train,y_train, batch_size=16,epochs=500, validation_split = 0.25, callbacks = [ers], class_weight = weights)
    return model


def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(x_LOCATION)
    blob.upload_from_filename('X_test_500.npy')
    blob = bucket.blob(y_LOCATION)
    blob.upload_from_filename('y_test_500.npy')
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('model_500.h5')


def save_model(model, X_test, y_test):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    np.save('X_test_500.npy', X_test)
    np.save('y_test_500.npy', y_test)

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    model.save("model_500.h5")
    print("Saved model to disk")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")
    

if __name__ == '__main__':
    # get training data from GCP bucket
    X_train, X_test, y_train, y_test = get_data()

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    model = train_model(X_train, y_train)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(model, X_test, y_test)
    
