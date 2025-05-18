import faiss
import numpy as np
import pickle
import os

VECTOR_DB_PATH = "vector_db"
INDEX_PATH = os.path.join(VECTOR_DB_PATH, "faiss.index")
CHUNKS_PATH = os.path.join(VECTOR_DB_PATH, "chunks.pkl")

def save_vector_store(embeddings, chunks):
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_vector_store():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("Vector store files missing.")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
