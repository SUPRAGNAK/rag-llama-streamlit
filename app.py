import streamlit as st
from rag_chain import retrieve_relevant_chunks, query_llama_with_context
from vector_store import load_vector_store

st.set_page_config(page_title="RAG Chat", layout="wide")

# Load FAISS index and chunks
index, chunks = load_vector_store()

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear chat button
col1, col2 = st.columns([0.85, 0.15])
with col2:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

# Title and intro
st.title("📚 Local RAG Chat with LLaMA 3 (Ollama)")
st.markdown("Chat over your local PDF & DOCX files with contextual memory")
st.divider()

# Chat history
chat_container = st.container()
with chat_container:
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Assistant:** {chat['bot']}")
            st.divider()
    else:
        st.info("No chat yet. Ask your first question below!")

# Input box
with st.form("chat_form", clear_on_submit=True):
    query = st.text_input("Ask your question", key="input_text")
    submitted = st.form_submit_button("Send")

if submitted and query:
    with st.spinner("Thinking..."):
        top_chunks = retrieve_relevant_chunks(query, index, chunks)
        response = query_llama_with_context(query, top_chunks)

    st.session_state.chat_history.append({
        "user": query,
        "bot": response
    })