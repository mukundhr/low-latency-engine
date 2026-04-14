from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")


class Embedder:
    def __init__(self, model_name=None):
        self.model_name = model_name or DEFAULT_MODEL
        self.model = SentenceTransformer(self.model_name, device="cuda")

    def encode(self, text):
        return self.model.encode(text, normalize_embeddings=True)