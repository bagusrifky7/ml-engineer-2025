from fastapi import FastAPI
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# create endpoint
app = FastAPI()


# load pipeline
with open('encode_pipeline.pkl', 'rb') as file_encode:
    encode = pickle.load(file_encode)

# load model
model = load_model('fix-model.keras')

# get for returning health status OK
@app.get('/health')
def health():
    return {
        'status': 'OK'
    }

# post to serve prediction
@app.post('/predict')
def predict(features: dict):
    # turn dictionary into dataframe
    df = pd.DataFrame([features])
    
    # encode dataframe filled with features
    df_encode = encode.transform(df)

    # prediction process
    y_inf_prob = model.predict(df_encode)[0][0]

    # we want probability more than 0.5 and make it integer not float
    y_inf = int(y_inf_prob > 0.5)

    return {
        'Prediksi Kemiskinan': y_inf,
        'Probabilitas Prediksi': y_inf_prob
    }


