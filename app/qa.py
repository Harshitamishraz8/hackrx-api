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
            context_chunks = query_pinecone(question, top_k=3)
            context = "\n".join(context_chunks)[:3000]  # truncate for LLM if needed
            
            prompt = f"""You are an expert insurance policy analyst. Answer the following user question using ONLY the relevant information from the provided insurance policy context.

Context: {context}

Question: {question}

Instructions:
- Only answer what is asked.
- If answer is not in context, say "Information not available in the document."
- Avoid unnecessary details or extra policy info.
- Keep it short and to the point.

Answer:"""

            chat_completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert insurance policy analyst. Provide accurate, specific answers based on policy documents."},
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

    return { "answers": all_answers }
