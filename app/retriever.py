import faiss
import pickle
import numpy as np

FAISS_INDEX_PATH = "data/index/faiss.index"
DATA_PATH = "data/processed/data.pkl"


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        with open(DATA_PATH, "rb") as f:
            data = pickle.load(f)

        self.chunks = data["chunks"]
        self.metadata = data["metadata"]

    def search(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding]).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "chunk_id": self.metadata[idx]["chunk_id"],
                    "score": float(scores[0][i])
                })

        return results