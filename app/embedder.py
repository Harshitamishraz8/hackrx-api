# app/embedder.py
from sentence_transformers import SentenceTransformer

# Load 384-dim model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return model.encode(text).tolist()
