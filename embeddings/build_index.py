import faiss
import pickle
from sentence_transformers import SentenceTransformer
from ingestion.process_docs import process_pdf


def build_index_for_pdf(pdf_path):
    # 1. Chunk the PDF
    chunks = process_pdf(pdf_path)

    # 2. Create embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts)

    # 3. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks
