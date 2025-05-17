import faiss
import numpy as np
import pickle

FAISS_INDEX_PATH = "faiss_index.index"
CHUNKS_PATH = "chunks.pkl"

def save_vector_store(embeddings, chunks):
    embeddings_array = np.array(embeddings).astype('float32')
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_vector_store():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks