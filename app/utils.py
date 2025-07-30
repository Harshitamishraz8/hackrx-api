# app/utils.py
from typing import List
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import time
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

def upsert_to_pinecone(vectors: List[dict], namespace: str = "default"):
    try:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            print(f"✅ Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            time.sleep(0.1)
    except Exception as e:
        print(f"❌ Error upserting to Pinecone: {e}")
        raise

def query_pinecone(query_text: str, document_id: str, top_k: int = 5, namespace: str = "default"):
    try:
        print(f"🔍 Encoding query: {query_text}")
        embedding = model.encode(query_text).tolist()

        print(f"📡 Querying Pinecone for doc_id: {document_id}")
        response = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter={
                "document_url": {"$contains": document_id}  # Or exact match depending on your metadata
            }
        )
        return response.matches
    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        return []
