from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from app.qa import get_answer
from app.models import QARequest
import os

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
    expected_token = os.getenv("API_AUTH_TOKEN", "d809808918dd2a7d6b11fa5b23fa01e3abf9814dd225582d4d5674dc2138be0b")
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials.credentials

@app.post("/hackrx/run", tags=["Run Qa"])
async def run_qa(request: QARequest, token: str = Depends(verify_token)):
    try:
        # Process the document and get answers
        answers = get_answer(request.questions, request.documents)
        
        # Return in the exact format expected by the platform
        return {"answers": [item["answer"] for item in answers]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/", tags=["Health Check"])
async def health_check():
    return {"status": "API is running", "message": "HackRx Document Q&A API"}