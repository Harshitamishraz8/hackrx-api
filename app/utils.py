# app/utils.py
from typing import List
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

def upsert_to_pinecone(vectors: List[dict], namespace: str = "default"):
    index.upsert(vectors=vectors, namespace=namespace)
