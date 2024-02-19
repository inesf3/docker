from sentence_transformers import SentenceTransformer, util

class Model:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def predict(self, sentence1 : str, sentence2 : str) -> float:
        embeddings1 = self.model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentence2, convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings1, embeddings2)
        return cosine_score.item()