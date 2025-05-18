import requests

def query_llama_with_context(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""Use the following context to answer the question. When the right context is not provided please reply with a apt message wherever required.

Context:
{context}

Question: {query}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )

    return response.json().get("response", "").strip()
