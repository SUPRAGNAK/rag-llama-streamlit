# rag_chain.py

from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def query_llama_with_context(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""Use the following context to answer the question.
If the context is not sufficient, say that politely.

Context:
{context}

Question: {query}

Answer:"""

#    response = client.chat.completions.create(
#    response = openai.ChatCompletion.create(
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You will repond to the the context, if required details are not available provide basic informtation through the genAI capabilities and do not hallucnate on the content created."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()
