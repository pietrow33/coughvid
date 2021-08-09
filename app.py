import streamlit as st
import tensorflow as tf
import librosa
from librosa import display
import matplotlib.pyplot as plt
import imageio
import numpy as np
from PIL import Image
from tensorflow.keras.backend import expand_dims

uploaded_file = st.file_uploader("Upload File")

model = tf.keras.models.load_model('./notebooks/models_coughvid_model.h5')

def transform_audio(audio):
    
    audio, rate = librosa.load(audio)
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

X = transform_audio(uploaded_file)

y = model.predict(X)

print(y)

st.image(X)

st.audio(uploaded_file)

st.write(y)