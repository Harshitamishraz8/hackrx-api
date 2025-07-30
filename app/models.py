# app/models.py
from typing import List
from pydantic import BaseModel

class QARequest(BaseModel):
    documents: str
    questions: List[str]
