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


uploaded_file = st.file_uploader("Upload File")

model = tf.keras.models.load_model('./notebooks/models_coughvid_model.h5')

loaded_model = pickle.load(open(os.path.join('./models', 'cough_classifier'), 'rb'))
loaded_scaler = pickle.load(open(os.path.join('./models','cough_classification_scaler'), 'rb'))

audio, rate = librosa.load(uploaded_file)

def cough_detect(audio, rate):
    
    probability = classify_cough(audio, rate, loaded_model, loaded_scaler)
    
    return probability

def transform_audio(audio, rate):
    
    S = librosa.feature.melspectrogram(y = audio, sr = rate, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=rate, fmax=8000, ax=ax)
    ax.axis("off")

    plt.savefig('./front_end_pics/user8.png', bbox_inches='tight', pad_inches = 0)
    
    plt.close()
    
    img2 = Image.open('./front_end_pics/user8.png')
    img2 = img2.resize((334, 217),Image.ANTIALIAS)
    
    img2.save('./front_end_pics/user9.png',optimize=True,quality=95)
    
    X = imageio.imread('./front_end_pics/user9.png')
    
    X = np.array(expand_dims(X, axis=0))
    
    return X

probability = cough_detect(audio, rate)

if probability <= 0.6:
    st.write("Cough again bitch!")
    st.write(probability)
else:
    X = transform_audio(audio, rate)

    y = model.predict(X)

    print(y)

    st.image(X)

    st.audio(uploaded_file)

    st.write(y)

    st.write(probability)
    