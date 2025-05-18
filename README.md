# RAG Chat with LLaMA 3 (Ollama)

This is a local Retrieval-Augmented Generation (RAG) chat app powered by LLaMA 3 via Ollama, with support for PDF and DOCX document ingestion and FAISS vector store retrieval.

## Features
- Upload and parse local PDF & DOCX files
- Store embeddings using FAISS
- Ask questions using context-aware LLM responses
- Local LLaMA 3 via Ollama

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
