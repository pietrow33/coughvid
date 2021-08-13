from re import U
from librosa.core.audio import load
import streamlit as st
import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
from coughvid.DSP import classify_cough
import os
import subprocess
import streamlit as st
import base64
import json
import ast
import shlex
import subprocess
from coughvid.segmentation import segment_cough

st.set_page_config(
    page_title="Coughvid", # => Quick reference - Streamlit
    page_icon="😷",
    layout="centered", # wide
    initial_sidebar_state="auto",) # collapsed

@st.cache
def load_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return encoded

def image_tag(path):
    encoded = load_image(path)
    tag = f'<img src="data:image/png;base64,{encoded}">'
    return tag

def background_image_style(path):
    encoded = load_image(path)
    style = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    '''
    return style

image_path = 'tempDir/back4.jpg'
image_link = 'https://republicanos10.org.br/wp-content/uploads/2017/06/worldwide-\
    connection-background-ipad-wacom-abstract-images-cool-white-background.jpg'

st.write(background_image_style(image_path), unsafe_allow_html=True)

st.image("./tempDir/logo4.png", width=250,)

st.title("This application provides an imediate COVID-19 diagnosis through cough audio analysis.")
    
st.subheader("Please, drag and drop an audio file to get tested.")
             
             
def cough_detect(audio, rate):
    probability = classify_cough(audio, rate, loaded_model, loaded_scaler)
    return probability


def audios_trim(audios, rates):

    cough_segments, cough_mask = segment_cough(audios,rates)
    if len(cough_segments) == 0:
        trim = np.zeros(1)
    else:
        trim = np.concatenate(cough_segments)
    audio_size = trim.shape[0]/rates
    
    print("ok aqui")
    print(audio_size)
    print(trim.shape)
    
    return trim, audio_size


def post_process(trim, audio_size, max_audio_size, rate):

    if audio_size < max_audio_size:
        pad = int(((rate*max_audio_size - trim.shape[0]) // 2))
        print("ok ----------")
        print(pad)
        if ((rate*max_audio_size - trim.shape[0]) % 2 == 0):
            new_trim = np.concatenate((np.zeros(pad),trim,np.zeros(pad)))
        else:
            new_trim = np.concatenate((np.zeros(pad+1),trim,np.zeros(pad)))
    elif audio_size > max_audio_size:
        pad = int(((rate*max_audio_size - trim.shape[0]) // 2))
        arr = trim
        if ((rate*max_audio_size - trim.shape[0]) % 2 == 0):
            new_trim = (arr[-pad:pad])
        else:
            new_trim = (arr[-pad-1:pad])  
    else:
        new_trim = trim
    print("foi ------")
    print(new_trim.shape)
    return new_trim


# def transform_audio(audio, rate):

#     S = librosa.feature.melspectrogram(y = audio, sr = rate, n_mels=128, fmax=8000)
#     fig, ax = plt.subplots()
#     S_dB = librosa.power_to_db(S, ref=np.max)
#     img = display.specshow(S_dB, sr=rate, fmax=8000, ax=ax)
#     ax.axis("off")

#     plt.savefig('./tempDir/user.png', bbox_inches='tight', pad_inches = 0)
#     plt.close()

#     img2 = Image.open('./tempDir/user.png')
#     img2 = img2.resize((334,217),Image.ANTIALIAS)
#     img2.save('user2.png',optimize=True,quality=95)
#     st.write("This is your cough's mel spectogram!")
#     st.image(img2, use_column_width=True)
    
#     X = imageio.imread('user2.png')
#     print("esse foi")
#     X = X / 255.
#     X = np.array(X[np.newaxis])
   
#     model = load_model("./models/models_coughvid_model.h5")
#     print("esse foi também")
    
#     y = model.predict(X)
    
#     resultado = float(np.round(y[0][0], 0))

#     return resultado
             

#Recieve the loaded file from streamlit
uploaded_file = st.file_uploader("Upload File")

if uploaded_file:
     #Get file type, save and transform it to .wav if not already, then read audio
    file_type = str(uploaded_file)
    file_type = file_type[file_type.find("type")-7:file_type.find("type")-3]
    if file_type != ".wav":
        st.warning("Sorry, we are only accepting .wav files for now. Please, upload again.")
    else:
        audio, rate = librosa.load(uploaded_file, sr=None)
        
        loaded_model = pickle.load(open(os.path.join('./models', 'cough_classifier'), 'rb'))
        loaded_scaler = pickle.load(open(os.path.join('./models','cough_classification_scaler'), 'rb'))

        #Call cough detection function to get the probability of audio having cough
        probability = cough_detect(audio, rate)

        #Asks for new audio if cough detection level is below 60%, else predicts with the model to the given audio
        if probability <= 0.6:
            st.audio(uploaded_file)
            st.warning("Please record another audio. No cough detected.")
        else:
            # resultado = transform_audio(audio, rate)
            trim, audio_size = audios_trim(audio, rate)
            new_trim = post_process(trim, audio_size, max_audio_size = 3, rate = rate)
            
            S = librosa.feature.melspectrogram(y = new_trim, sr = rate, n_mels=128, fmax=8000)
            fig, ax = plt.subplots()
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = display.specshow(S_dB, sr=rate, fmax=8000, ax=ax)
            ax.axis("off")
            plt.savefig('./tempDir/user.png', bbox_inches='tight', pad_inches = 0)
            plt.close()
            
            img2 = Image.open('./tempDir/user.png')
            img2 = img2.resize((223, 163),Image.ANTIALIAS)
            img2.save('user2.png',optimize=True,quality=95)
            
            fig, ax = plt.subplots()
            img = display.specshow(S_dB, x_axis='time', y_axis='mel', sr=rate, fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            plt.savefig('./tempDir/user.png', transparent = True)
            plt.close()
        
            st.write("This is your cough's Mel spectogram! A spectrogram is a visualization of the frequency \
            spectrum of a signal, which is the frequency range that is contained by the signal. \
                The Mel scale mimics how the human ear works. The Mel spectrogram is a spectrogram that is converted to a Mel scale. ")
            st.image('./tempDir/user.png', use_column_width=True)
            
            #https://coughvid-test-cs2qj3qyha-ew.a.run.app/predict
            
            cmd = "curl -X 'POST' 'https://coughvid-api-cs2qj3qyha-ew.a.run.app/predict' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' \
            -F 'file=@user2.png;type=image/png'"
            
            args = shlex.split(cmd)
            process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            os.remove("./tempDir/user.png")
            os.remove("user2.png")

            my_json = stdout.decode('utf8').replace("'", '"')
            print(my_json)
            print('- ' * 20)

            # Load the JSON to a Python list & dump it back out as formatted JSON
            data = json.loads(my_json)
            response = json.dumps(data, indent=4, sort_keys=True)
            response = ast.literal_eval(response)
            print(response)
            
            # resultado = resultado['pred']
        
            st.audio(uploaded_file)
            
            if response['pred'] == 1:
                st.write(f"Test result: COVID-19 positive")
                st.error("There is a great chance of you being COVID-19 positive. However, our model is not \
                    100% accurate. If you are experiencing symptoms, we recommend that you get tested.")
            else :
                st.write(f"Test result: Healthy")
                st.success("Nice! The chance of you being COVID-19 positive is low. However, our model is not \
                    100% accurate. If you are experiencing symptoms, we recommend that you get tested.")
                st.balloons()


st.write('''If you want to learn more, check out the side bar.''') #\n
# Also, we are sharing our model\'s confusion matrix, if you are curious, click below to check it out!''')
# if st.button('click here!'):
#     # print is visible in server output, not in the page
#     st.write('Confusion Matrix')
#     st.image('tempDir/confusion.png', use_column_width=True)
#     st.write("Nice isn't it?")



#if st.sidebar.button('Click for a deepper techinical dive into our approach.'):
    # print is visible in server output, not in the page
#    print('button clicked!')
st.sidebar.markdown('''
# COVID-19 Instantaneous Detection through cough audio analysis.

## A deepper dive into our approach.

Like other respiratory illnesses, COVID-19 major symptoms include breathing problems and cough.\
Previous studies on respiratory conditions suggest that sound detection programs can be used to identify \
other respiratory diseases such as asthma . This technique has been now employed in COVID-19 detection through \
data analyses all over the world.

In 2020, the Lausanne University collected a large sample of cough records from both healthy \
and covid positive tested volunteers. This open-source data composes the aim of this analysis. 

While Audio Classification is still an uncommon field in data science, the technology to represent sounds as \
images through Mel Spectrograms (Mel scale) exists from the 1930’s. Recently, this topic gained attention \
because of the COVID-19 pandemic, once it can detect different audio frequencies with a high level of accuracy, \
making possible feature extraction from cough record samples.

Using those records and transforming them in features allowed to create and train a model capable of classifying \
a subject as COVID-19 positive or healthy. The model gave origin to an application where users can upload audio cough \
samples and receive an instantaneous result: COVID-19 positive or negative.

¹DINKO Oletic, BILAS, Vedran, Energy-efficient respiratory sounds sensing for personal mobile asthma monitoring,\
IEEE Sensors Journal 16, 23 (Dec. 2016).

²https://github.com/virufy/virufy-data

³Librosa Documentation
    
    ''')

authors = '<footer class="css-8xv65a eknhn3m4"><p>Authors: <br> <p>Manuela - manuelatgknegt@hotmail.com\
    <br>Pietro - pietrow.pw@gmail.com <br>Rodrigo - rodrigovgoulart@gmail.com</a></footer>'

st.write(authors, unsafe_allow_html=True)