import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key")

def generate_answer(context: str, question: str) -> str:
    prompt = f"""You are an expert assistant. Use the given context to answer the user's question strictly from the document.

Context:
{context}

Question: {question}
Answer:"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            },
            timeout=20
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Failed to generate answer due to: {str(e)}"
