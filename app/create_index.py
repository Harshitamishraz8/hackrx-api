# create_index.py
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from pinecone import ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")
environment = os.getenv("PINECONE_ENV", "gcp-starter")

try:
    existing_indexes = [i['name'] for i in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating index {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="gcp",
                region="us-central1"
            )
        )
        print(f"✅ Created index {index_name} with dimension 384")
    else:
        print("ℹ️ Index already exists")
        
    # Test the index
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"📊 Index stats: {stats}")
    
except Exception as e:
    print(f"❌ Error with Pinecone index: {e}")
    print("Please check your Pinecone credentials and try again.")
