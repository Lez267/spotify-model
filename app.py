from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title='Spotify Popularity Prediction')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = load(pathlib.Path('model/spotify-model-v1.joblib'))

class InputData(BaseModel):
    danceability: float = 0.651
    energy: float = 0.800
    key: float = 1.0
    loudness: float = -5.359
    mode: float = 1.0
    speechiness: float = 0.1
    acousticness: float = 0.2
    instrumentalness: float = 0.0
    liveness: float = 0.1
    valence: float = 0.5
    tempo: float = 120.0
    duration_ms: float = 200000.0

class OutputData(BaseModel):
    popularidad: float = 0.0
    score: float = 0.0

@app.get('/')
def root():
    return {
        'message': 'Spotify Popularity Prediction Model API is running',
        'status': 'online',
        'endpoints': {
            'docs': '/docs',
            'score': 'POST /score'
        }
    }

@app.post('/score', response_model=OutputData)
def score(data: InputData):
    model_input = np.array([v for k, v in data.dict().items()]).reshape(1, -1)
    result = float(model.predict(model_input)[0])
    return {
        'popularidad': round(result, 2),
        'score': round(result, 2)
    }
