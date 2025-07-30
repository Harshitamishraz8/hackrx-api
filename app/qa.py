# app/qa.py

from app.retriever import query_pinecone
from app.ingest import process_and_store_pdf
from app.llm import run_llm

def get_answer(questions: list[str], document_url: str):
    try:
        process_and_store_pdf(document_url)
    except Exception as e:
        print(f"Warning: Error processing document: {e}")
        # Proceed with whatever embeddings are available

    all_answers = []
    for question in questions:
        try:
            context_chunks = query_pinecone(question)
            context = "\n".join(context_chunks)

            answer = run_llm(context, question)
            all_answers.append({"question": question, "answer": answer})
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            all_answers.append({"question": question, "answer": "Error processing this question. Please try again."})

    return all_answers
