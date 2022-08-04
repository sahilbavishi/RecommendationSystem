import logging
import joblib
import sys
import pandas as pd
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from datetime import datetime
import pickle
from PyFile import recc

app = FastAPI()

pickle_in = open('recc.pkl', 'rb')
a = pickle.load(pickle_in)

@app.get('/status/')
def HealthCheck():
    return 'HI'

class RequestBody(BaseModel):
    movie_name : str;
    userID : int

@app.post('/api/recommendations/')
def predict(request_body: RequestBody):
    movie_name = request_body.movie_name
    userID = request_body.userID
    return(a.get_recommendations(movie_name, 10))