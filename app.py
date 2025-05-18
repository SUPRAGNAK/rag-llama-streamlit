# app.py

import streamlit as st
from rag_chain import query_llama_with_context
from vector_store import load_vector_store
from retriever import get_top_k_chunks

st.set_page_config(page_title="RAG Chat with OpenAI", layout="wide")

# Load vector store and chunks
vector_store, chunks = load_vector_store()

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "input_submitted" not in st.session_state:
    st.session_state.input_submitted = False
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# --- Title ---
st.title("📚 Local RAG Chat with OpenAI")
st.markdown("Chat with your PDF/DOCX files using OpenAI + FAISS.")
st.divider()

# --- Clear Chat Button ---
col1, col2 = st.columns([0.85, 0.15])
with col2:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- Chat History ---
st.markdown("### Chat History")
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['bot']}")
        st.divider()
else:
    st.info("No chat yet. Ask your first question below!")

# --- Input Form ---
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask your question",
        key="input_text",
        placeholder="Type your question and press Enter...",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Submit")

# --- Handle Input ---
if submitted and user_input.strip():
    with st.spinner("🤖 Assistant is thinking..."):
        top_chunks = get_top_k_chunks(user_input, vector_store, chunks)
        response = query_llama_with_context(
            user_input,
            top_chunks,
            st.session_state.chat_history
        )

    st.session_state.chat_history.append({
        "user": user_input,
        "bot": response
    })

    # Limit memory to last 15 messages
    if len(st.session_state.chat_history) > 15:
        st.session_state.chat_history = st.session_state.chat_history[-15:]

    st.rerun()