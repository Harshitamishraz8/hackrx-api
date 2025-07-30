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
    try:
        print(f"Querying Pinecone for: {question}")
        query_vector = embed_model.encode(question).tolist()
        
        response = index.query(
            vector=query_vector, 
            top_k=8, 
            include_metadata=True,
            include_values=False
        )
        
        if not response['matches']:
            print("No matches found in Pinecone")
            return ["No relevant information found in the knowledge base."]
        
        # Sort by score and return top matches
        matches = sorted(response['matches'], key=lambda x: x['score'], reverse=True)
        contexts = []
        
        for match in matches:
            if match['score'] > 0.7:  # Only include high-confidence matches
                contexts.append(match['metadata']['text'])
        
        if not contexts:
            return ["No highly relevant information found for this question."]
            
        print(f"Found {len(contexts)} relevant contexts")
        return contexts[:5]  # Return top 5 most relevant contexts
        
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return ["Error retrieving information from knowledge base."]
