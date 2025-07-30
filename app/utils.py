# app/utils.py
from typing import List
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import time

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

def upsert_to_pinecone(vectors: List[dict], namespace: str = "default"):
    try:
        # Upsert in batches to avoid rate limits
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            print(f"Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            time.sleep(0.1)  # Small delay to avoid rate limits
        
        print(f"Successfully upserted {len(vectors)} vectors to Pinecone")
    except Exception as e:
        print(f"Error upserting to Pinecone: {e}")
        raise
