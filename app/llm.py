# app/llm.py

import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"  # Or "mixtral-8x7b-32768"

def run_llm(context: str, question: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in environment variables.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""You are an expert insurance and legal document analyst. Answer the question based on the provided context from insurance policy documents.

Context: {context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context.
- If the information is not available in the context, state "I don't know" or "Information not available in the provided documents".
- Be specific about policy terms, waiting periods, coverage limits, and conditions.
- Use exact terminology from the policy documents when possible.

Answer:"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert insurance policy analyst. Provide accurate, specific answers based on policy documents."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()
