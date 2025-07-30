from app.embed import embed_text, embed_query
from app.utils import extract_text_from_url
from app.pinecone_client import get_pinecone_index
from app.llm import generate_answer

def get_answer(questions, document_url):
    # Step 1: Extract text from PDF
    full_text = extract_text_from_url(document_url)
    
    # Step 2: Split into chunks (you may improve chunking if needed)
    chunks = full_text.split('\n\n')  # crude chunking

    # Step 3: Embed and upsert into Pinecone
    embeddings = embed_text(chunks)
    index = get_pinecone_index()

    namespace = "document"  # You may want to make this dynamic per request
    index.delete(delete_all=True, namespace=namespace)

    to_upsert = [(f"id-{i}", vec, {"text": chunk}) for i, (vec, chunk) in enumerate(zip(embeddings, chunks))]
    index.upsert(vectors=to_upsert, namespace=namespace)

    # Step 4: Answer each question individually with relevant context
    answers = []
    for q in questions:
        q_embed = embed_query(q)
        res = index.query(vector=q_embed, top_k=5, include_metadata=True, namespace=namespace)
        
        contexts = [match['metadata']['text'] for match in res['matches']]
        context_str = "\n\n".join(contexts)

        answer = generate_answer(context_str, q)
        answers.append({"question": q, "answer": answer.strip()})

    return answers
