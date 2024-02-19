# content of test_sample.py
import time
from similarity_model import Model
model = Model()

def read_root(sentence1: str, sentence2: str):
    start = time.time()
    similarity_score = model.predict(sentence1, sentence2)
    end = time.time()

    return {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "similarity_score": similarity_score,
        "time_taken": end - start
    } 

# test read_root function
def test_read_root():
    response = read_root("I am a sentence.", "I am also a sentence.")
    assert type(response["similarity_score"]) == float
    assert type(response["time_taken"]) == float
    assert response["sentence1"] == "I am a sentence."
    assert response["sentence2"] == "I am also a sentence."