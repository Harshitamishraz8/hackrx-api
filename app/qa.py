# app/qa.py

import os
from app.retriever import query_pinecone
from app.ingest import process_and_store_pdf
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_answer(questions: list[str], document_url: str):
    try:
        # Ingest and store document embeddings
        process_and_store_pdf(document_url)
    except Exception as e:
        print(f"Warning: Error processing document: {e}")
        # Proceed anyway if already indexed

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    all_answers = []

    for question in questions:
        try:
            # Fetch relevant chunks for this specific question
            context_chunks = query_pinecone(question)
            context = "\n".join(context_chunks)

            # Individualized prompt for each question
            prompt = f"""
You are an expert legal and insurance policy analyst.

Context:
{context}

Question:
{question}

Instructions:
- Answer only based on the context provided.
- If the answer is not in the context, reply: "Information not available in the provided documents."
- Be precise, concise, and use insurance/legal terminology.

Answer:"""

            chat_completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful insurance policy assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )

            answer = chat_completion.choices[0].message.content.strip()
            all_answers.append({"question": question, "answer": answer})

        except Exception as e:
            print(f"Error with question '{question}': {e}")
            all_answers.append({
                "question": question,
                "answer": "Error processing this question. Please try again."
            })

    return all_answers
