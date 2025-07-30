from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils.pdf_loader import load_and_split_pdf
from utils.vector_store import get_relevant_chunks
from utils.llm import get_answers
import os

app = FastAPI()
TOKEN = os.getenv("HACKRX_TOKEN", "hackrx-secret-token")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def run_query(request: Request, body: QueryRequest):
    auth = request.headers.get("Authorization")
    if auth != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    
    try:
        chunks = load_and_split_pdf(body.documents)
        results = []
        for q in body.questions:
            top_chunks = get_relevant_chunks(chunks, q)
            answer = get_answers(q, top_chunks)
            results.append(answer)
        return {"answers": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
