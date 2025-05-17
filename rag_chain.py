from embedder import get_embedding
import numpy as np
import requests

def retrieve_relevant_chunks(query, index, chunks, k=3):
    query_embedding = get_embedding(query)
    query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)
    D, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0]]

def query_llama_with_context(query, chunks):
    context = "\n".join(chunks)
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    return response.json()["response"].strip()