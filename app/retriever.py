# app/retriever.py
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

def query_pinecone(question: str):
    query_vector = embed_model.encode(question).tolist()
    response = index.query(vector=query_vector, top_k=5, include_metadata=True)
    return [match['metadata']['text'] for match in response['matches']]
