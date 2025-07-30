# app/ingest.py
import os
import tempfile
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from app.utils import upsert_to_pinecone
from dotenv import load_dotenv
import hashlib
import time

load_dotenv()
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

def process_and_store_pdf(pdf_url: str):
    try:
        print(f"Processing PDF from URL: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        raise
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        all_text = ""
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    all_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            except Exception as e:
                print(f"Error extracting text from page {page_num + 1}: {e}")
                continue
        
        if not all_text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        print(f"Extracted {len(all_text)} characters from PDF")
        
        # Create overlapping chunks for better context
        chunk_size = 800
        overlap = 200
        chunks = []
        
        for i in range(0, len(all_text), chunk_size - overlap):
            chunk = all_text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        print(f"Created {len(chunks)} chunks")
        
        # Create unique document ID based on URL
        doc_id = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
        
        vectors = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = model.encode(chunk).tolist()
                vectors.append({
                    "id": f"{doc_id}-chunk-{i}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "document_url": pdf_url,
                        "chunk_index": i,
                        "timestamp": int(time.time())
                    }
                })
            except Exception as e:
                print(f"Error creating embedding for chunk {i}: {e}")
                continue
        
        if vectors:
            upsert_to_pinecone(vectors)
            print(f"Successfully stored {len(vectors)} vectors in Pinecone")
        else:
            raise ValueError("No vectors were created from the document")
            
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

