import streamlit as st
import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
from coughvid.DSP import classify_cough
# import os
import subprocess
import streamlit as st
# import base64
import json
import ast
import shlex
import subprocess

st.header('CoughVid')
st.subheader('Este aplicativo dá um diagnóstico imédiato de COVID-19 pelo som da tosse. Por favor, clique no botão abaixo para gravar sua tosse. Em seguida clique em enviar')

#Recieve the loaded file from streamlit
uploaded_file = st.file_uploader("Upload File")

if uploaded_file:
    
    #Get file type, save and transform it to .wav if not already, then read audio
    # file_type = str(uploaded_file)
    # file_type = file_type[file_type.find("type")-7:file_type.find("type")-3]
    # if file_type != ".wav":
    #     with open(os.path.join("tempDir",f"test{file_type}"),"wb") as f: 
    #         f.write(uploaded_file.getbuffer())
    #     subprocess.call(["ffmpeg", "-i",f"./tempDir/test{file_type}", "./tempDir/test.wav"])
    #     audio, rate = librosa.load("./tempDir/test.wav", sr=None)
    #     os.remove(f"./tempDir/test{file_type}")
    #     os.remove("./tempDir/test.wav")
    # else:
    #     audio, rate = librosa.load(uploaded_file, sr=None)


    # loaded_model = pickle.load(open(os.path.join('./models', 'cough_classifier'), 'rb'))
    # loaded_scaler = pickle.load(open(os.path.join('./models','cough_classification_scaler'), 'rb'))

    def cough_detect(audio, rate):
        probability = classify_cough(audio, rate, loaded_model, loaded_scaler)
        return probability

    def transform_audio(audio, rate):
        
        S = librosa.feature.melspectrogram(y = audio, sr = rate, n_mels=128, fmax=8000)
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = display.specshow(S_dB, sr=rate, fmax=8000, ax=ax)
        ax.axis("off")

        plt.savefig('./tempDir/user.png', bbox_inches='tight', pad_inches = 0)
        plt.close()
        
        img2 = Image.open('./tempDir/user.png')
        img2 = img2.resize((334, 217),Image.ANTIALIAS)
        img2.save('user2.png',optimize=True,quality=95)
        st.image(img2)
        
        cmd = "curl -X 'POST' 'https://coughvid-test-cs2qj3qyha-ew.a.run.app/predict' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' \
            -F 'file=@user2.png;type=image/png'"
        args = shlex.split(cmd)
        process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        # os.remove("./tempDir/user.png")
        # os.remove("user2.png")

        my_json = stdout.decode('utf8').replace("'", '"')
        print(my_json)
        print('- ' * 20)

        # Load the JSON to a Python list & dump it back out as formatted JSON
        data = json.loads(my_json)
        response = json.dumps(data, indent=4, sort_keys=True)
        response = ast.literal_eval(response)
        print(response)
                
        return response

    #Call cough detection function to get the probability of audio having cough
    probability = cough_detect(audio, rate)

    #Asks for new audio if cough detection level is below 60%, else predicts with the model to the given audio
    if probability <= 0.6:
        st.write("Please record another audio. No cough detected.")
    else:
        resultado = transform_audio(audio, rate)
        print(f"ESSE AQUIIIII   {resultado}")
        print(type(resultado))
        
        resultado = resultado['pred']
    
        st.audio(uploaded_file)
        st.write(f"The probability of having COVID through Audio analysis is: {(resultado * 100)}%")
        st.write(f"Classification: {resultado}")
        
        if resultado == 1:
            ('Existe uma grande probabilidade de você estar com a COVID-19. Entretanto, nosso aplicativo não é 100% seguro. Recomendamos que você faça um teste de laboratório')
        else : 
            ('A probabilidade de você estar com a COVID-19 é baixa. Entretanto como nosso aplicativo não é 100% recomendamos que você faça um teste de laboratório')

else:
    st.write('Favor fazer o upload do arquivo de som.')


# -------------------------- PLANO B ---------------------------------
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# model = KerasClassifier(build_fn=create_model)
