from fastapi import FastAPI, File, UploadFile
import base64 
import shutil
import tensorflow as tf
import numpy as np
import imageio

app = FastAPI()

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open(f'21{file.filename}', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    X = imageio.imread(f'21{file.filename}')
    print("esse foi")
    X = X / 255.
    X = np.array(X[np.newaxis])

    
    model = tf.keras.models.load_model('models_coughvid_model.h5')
    print("esse foi também")
    
    y = model.predict(X)
    
    resultado = float(np.round(y[0][0], 0))
    
    print(f'ESSE É O PREDICT  - - - - ->  {resultado}')
    
    return {"pred": resultado}
#