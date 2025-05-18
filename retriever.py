import numpy as np
from embedder import get_embedding

def get_top_k_chunks(query, index, chunks, k=3):
    query_vec = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    D, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]
