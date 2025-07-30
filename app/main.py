from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from app.qa import get_answer
from app.ingest import process_and_store_pdf
from app.models import QARequest

app = FastAPI(
    title="HackRx Document Q&A API",
    version="0.1.0"
)

# Add CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Setup Bearer Auth for Swagger UI
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "hackrx-secret-token":
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials.credentials

@app.post("/hackrx/run", tags=["Run Qa"])
async def run_qa(request: QARequest, token: str = Depends(verify_token)):
    process_and_store_pdf(request.documents)
    answers = get_answer(request.questions, request.documents)
    return answers