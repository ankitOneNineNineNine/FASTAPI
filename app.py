import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
import os
import sys
from fastapi.logger import logger
from pydantic import BaseSettings

from fastapi.middleware.cors import CORSMiddleware

from sms_spam_detection import text_processing


class Settings(BaseSettings):
    # ... The rest of our FastAPI settings

    BASE_URL = "http://localhost:8000"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"


settings = Settings()


def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass


app = FastAPI()

if settings.USE_NGROK:
    # pyngrok should only ever be installed or initialized in a dev environment when this flag is set
    from pyngrok import ngrok

    # Get the dev server port (defaults to 8000 for Uvicorn, can be overridden with `--port`
    # when starting the server
    port = sys.argv[sys.argv.index(
        "--port") + 1] if "--port" in sys.argv else 8000

    # Open a ngrok tunnel to the dev server
    public_url = ngrok.connect(port).public_url
    logger.info(
        "ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

    # Update any base URLs or webhooks to use the public ngrok URL
    settings.BASE_URL = public_url
    init_webhooks(public_url)

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
# print(classifier)


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


PORT = os.getenv('PORT', 8000)
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=PORT)
