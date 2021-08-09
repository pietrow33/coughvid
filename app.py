import streamlit as st
import tensorflow as tf
import librosa
from librosa import display
import matplotlib.pyplot as plt
import imageio
import numpy as np
from PIL import Image
from tensorflow.keras.backend import expand_dims
import pickle
from coughvid.DSP import classify_cough
import os
import subprocess

#Recieve the loaded file from streamlit
uploaded_file = st.file_uploader("Upload File")

#Get file type, save and transform it to .wav if not already, then read audio
file_type = str(uploaded_file)
file_type = file_type[file_type.find("type")-7:file_type.find("type")-3]
if file_type != ".wav":
    with open(os.path.join("tempDir",f"test{file_type}"),"wb") as f: 
        f.write(uploaded_file.getbuffer())
    subprocess.call(["ffmpeg", "-i",f"./tempDir/test{file_type}", "./tempDir/test.wav"])
    audio, rate = librosa.load("./tempDir/test.wav", sr=None)
    os.remove(f"./tempDir/test{file_type}")
    os.remove("./tempDir/test.wav")
else:
    audio, rate = librosa.load(uploaded_file, sr=None)

#Load prediction model and cough detection model
model = tf.keras.models.load_model('./notebooks/models_coughvid_model.h5')
loaded_model = pickle.load(open(os.path.join('./models', 'cough_classifier'), 'rb'))
loaded_scaler = pickle.load(open(os.path.join('./models','cough_classification_scaler'), 'rb'))

def cough_detect(audio, rate):
    probability = classify_cough(audio, rate, loaded_model, loaded_scaler)
    return probability

def transform_audio(audio, rate):
    
    S = librosa.feature.melspectrogram(y = audio, sr = rate, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=rate, fmax=8000, ax=ax)
    ax.axis("off")

    plt.savefig('./tempDir/user.png', bbox_inches='tight', pad_inches = 0)
    plt.close()
    
    img2 = Image.open('./tempDir/user.png')
    img2 = img2.resize((334, 217),Image.ANTIALIAS)
    img2.save('./tempDir/user2.png',optimize=True,quality=95)
    os.remove("./tempDir/user.png")    

    X = imageio.imread('./tempDir/user2.png')
    X = X / 255.
    os.remove("./tempDir/user2.png")

    X = np.array(expand_dims(X, axis=0))
    return X

#Call cough detection function to get the probability of audio having cough
probability = cough_detect(audio, rate)

#Asks for new audio if cough detection level is below 60%, else predicts with the model to the given audio
if probability <= 0.6:
    st.write("Please record another audio. No cough detected.")
else:
    X = transform_audio(audio, rate)
    y = model.predict(X)
    print(y)

    st.image(X)
    st.audio(uploaded_file)
    st.write(f"The probability of having COVID through Audio analysis is: {(y[0][0]*100):.2f}%")
    st.write(f"Classification: {np.round(y[0][0],0)}")
