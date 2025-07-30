import os
from sentence_transformers import SentenceTransformer
import pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-index")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_relevant_chunks(text, questions):
    question_embeddings = model.encode(questions).tolist()
    results = index.query(vector=question_embeddings[0], top_k=5, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]
