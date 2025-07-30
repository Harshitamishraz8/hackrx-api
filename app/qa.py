# app/qa.py
import os
from app.retriever import query_pinecone
from app.ingest import process_and_store_pdf
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_answer(questions: list[str], document_url: str):
    try:
        process_and_store_pdf(document_url)
    except Exception as e:
        print(f"Warning: Error processing document: {e}")
        # Continue with existing knowledge base if document processing fails

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    all_answers = []
    for question in questions:
        try:
            context_chunks = query_pinecone(question)
            context = "\n".join(context_chunks)

            prompt = f"""
You are an expert insurance and legal document analyst. Answer the question based only on the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Provide a concise and accurate answer.
- If the information is not found in the context, respond with: "Information not available in the provided documents."
- Do not make assumptions or add explanations.
- Keep the answer policy-specific.

Answer:
"""

            chat_completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert insurance policy analyst. Provide accurate, concise answers based strictly on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            answer = chat_completion.choices[0].message.content.strip()
            all_answers.append({"question": question, "answer": answer})
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            all_answers.append({"question": question, "answer": "Error processing this question. Please try again."})

    return {"answers": all_answers}
