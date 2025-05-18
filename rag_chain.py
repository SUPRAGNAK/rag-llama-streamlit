# rag_chain.py

import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_llama_with_context(query, context_chunks, history=[]):
    context = "\n".join(context_chunks)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. You will repond to the the context, if required details are not available provide basic informtation through the genAI capabilities and do not hallucnate on the content created. Also try to remember the past chats and then provide relative information and expplanation"},
        {"role": "system", "content": f"Context:\n{context}"}
    ]

    # Add up to last 10–15 previous turns
    for pair in history[-10:]:
        messages.append({"role": "user", "content": pair["user"]})
        messages.append({"role": "assistant", "content": pair["bot"]})

    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" if available
        messages=messages
    )

    return response.choices[0].message.content.strip()