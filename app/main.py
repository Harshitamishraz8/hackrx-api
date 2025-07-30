from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from utils.pdf_utils import extract_text_from_url
from utils.pinecone_utils import get_relevant_chunks
from utils.llm import generate_answer
import os

app = FastAPI()

BEARER_TOKEN = os.getenv("HACKRX_API_TOKEN", "hackrx-secret-token")

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(req: Request, body: QueryRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {BEARER_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing token")

    doc_text = extract_text_from_url(body.documents)
    if not doc_text.strip():
        return {"answers": ["Information not available in the provided documents."] * len(body.questions)}

    chunks = get_relevant_chunks(doc_text, body.questions)
    answers = [generate_answer(question, chunks) for question in body.questions]
    return {"answers": answers}
