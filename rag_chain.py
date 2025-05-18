# rag_chain.py

import openai
import streamlit as st

# Read the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def query_llama_with_context(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""Use the following context to answer the question. 
If the context is not sufficient, say that politely.

Context:
{context}

Question: {query}

Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or use gpt-4 if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Based context replay with minimum reasoning and understanding then provide response."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()
