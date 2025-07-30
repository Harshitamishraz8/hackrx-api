# app/ingest.py
import os
import tempfile
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from app.utils import upsert_to_pinecone
from dotenv import load_dotenv

load_dotenv()
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))

def process_and_store_pdf(pdf_url: str):
    response = requests.get(pdf_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    all_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = [all_text[i:i+500] for i in range(0, len(all_text), 500)]

    vectors = [{
        "id": f"chunk-{i}",
        "values": model.encode(chunk).tolist(),
        "metadata": {"text": chunk}
    } for i, chunk in enumerate(chunks)]

    upsert_to_pinecone(vectors)
