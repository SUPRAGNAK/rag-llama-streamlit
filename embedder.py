from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text)

def embed_chunks(chunks):
    return [get_embedding(chunk) for chunk in chunks]
