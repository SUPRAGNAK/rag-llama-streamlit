# loader.py

from PyPDF2 import PdfReader
import docx
import os

def load_documents(folder_path):
    """
    Loads and concatenates text from all PDF and DOCX files in the given folder.
    """
    all_text = ""
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            try:
                reader = PdfReader(filepath)
                for page in reader.pages:
                    all_text += page.extract_text() or ""
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")
        elif filename.endswith(".docx"):
            try:
                doc = docx.Document(filepath)
                for para in doc.paragraphs:
                    all_text += para.text + "\n"
            except Exception as e:
                print(f"Error reading DOCX {filename}: {e}")
    return all_text

def split_text(text, chunk_size=500, overlap=50):
    """
    Splits the full text into overlapping chunks for embedding.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
