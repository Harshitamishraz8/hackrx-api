# app/llm.py

import os
import requests

def call_groq_llm(question, context):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""Answer the question using the context below. Be concise and accurate.
    
    Context:
    {context}

    Question: {question}
    Answer:"""

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a document assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "Failed to generate answer from the document."
