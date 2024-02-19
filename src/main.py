from .similarity_model import Model
from fastapi import FastAPI
import time


app = FastAPI()
model = Model()

@app.get("/")
def home():
    return {"Hello":"World"}

@app.get("/similarity")
def similarity(sentence1: str, sentence2: str):
    start = time.time()
    similar = model.predict(sentence1, sentence2)
    end = time.time()
    return {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "similarity": similar,
        "time_taken": end - start
        }
