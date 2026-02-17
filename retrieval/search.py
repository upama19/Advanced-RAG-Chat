import faiss
import pickle
from sentence_transformers import SentenceTransformer


class SemanticSearcher:
    def __init__(self, index_path, metadata_path):
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load chunk metadata
        with open(metadata_path, "rb") as f:
            self.chunks = pickle.load(f)

        # Load same embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query, top_k=5, active_document=None):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k * 3)

        results = []
        for idx in indices[0]:
            chunk = self.chunks[idx]

            if (
                active_document is None
                or chunk["metadata"]["source"] == active_document
            ):
                results.append(chunk)

            if len(results) == top_k:
                break

        return results


if __name__ == "__main__":
    searcher = SemanticSearcher(
        index_path="embeddings/faiss.index", metadata_path="embeddings/chunks.pkl"
    )

    query = "What is the U-Net architecture?"
    results = searcher.search(query)

    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(
            f"Source: {r['metadata']['source']} | Page: {r['metadata']['page_number']}"
        )
        print(r["text"][:500])
