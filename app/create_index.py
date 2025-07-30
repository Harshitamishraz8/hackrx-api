# create_index.py
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")

if index_name not in [i['name'] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec={"environment": os.getenv("PINECONE_ENV")}
    )
    print(f"✅ Created index {index_name} with dimension 384")
else:
    print("ℹ️ Index already exists")
