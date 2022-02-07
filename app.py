import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
import os

from fastapi.middleware.cors import CORSMiddleware

from sms_spam_detection import text_processing

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pickle_file = open("Spam_detection_NB.pkl", "rb")
classifier = pickle.load(pickle_file)
print(classifier)


@app.get('/')
def index():
    return {'message': "Hello World I am Prasanna"}


@app.post('/predict')
def predict_spamham(line):
    data = [line]
    value_out = classifier.predict(data)
    return{
        'Prediction':  value_out[0]
    }


PORT = os.getenv('PORT',8000)
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=PORT)
