# app/qa.py
import os
from app.retriever import query_pinecone
from app.ingest import process_and_store_pdf
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_answer(questions: list[str], document_url: str):
    process_and_store_pdf(document_url)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    all_answers = []
    for question in questions:
        context_chunks = query_pinecone(question)
        context = "\n".join(context_chunks)
        prompt = f"""
        Answer the question based on the context below. If the question can't be answered using the context, say "I don't know".

        Context: {context}
        Question: {question}

        Answer:
        """
        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = chat_completion.choices[0].message.content.strip()
        all_answers.append({"question": question, "answer": answer})

    return all_answers
