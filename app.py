import uvicorn
from fastapi import FastAPI
import numpy as np 
import pandas as pd 
import pickle

from sms_spam_detection import text_processing

app = FastAPI()

pickle_file = open("Spam_detection_NB.pkl","rb")
classifier = pickle.load(pickle_file)

@app.get('/')
def index():
	return {'message': "Hello World I am Prasanna"}

@app.post('/predict')
def predict_spamham(line):
	data =[line]
	value_out = classifier.predict(data)
	return{
		'Prediction':  value_out[0]
	}

# if __name__ == '__main__':
# 	uvicorn.run(app, host='127.0.0.0', port=8000)