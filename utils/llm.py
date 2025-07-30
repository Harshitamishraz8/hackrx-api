import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "mixtral-8x7b-32768"

def generate_answer(question, context_chunks):
    prompt = f"Answer the question based on context:\n\nContext:\n{''.join(context_chunks)}\n\nQuestion: {question}\nAnswer:"
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert document QA bot."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=25
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "Information not available due to an error."
