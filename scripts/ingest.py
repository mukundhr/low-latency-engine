import os
import sys
import time
import pickle
import hashlib

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from app.utils import chunk_text

# -------------------------
# PATHS
# -------------------------
PROCESSED_PATH = "data/processed/data.pkl"
FAISS_INDEX_PATH = "data/index/faiss.index"

MODEL_NAME = "BAAI/bge-small-en"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 40
BATCH_SIZE = 32


# -------------------------
# DEBUG START
# -------------------------
print("FILE STARTED:", __file__)


# -------------------------
# HASH (for duplicates)
# -------------------------
def file_hash(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


# -------------------------
# LOAD DOCUMENTS
# -------------------------
def load_documents(file_paths):
    docs = []
    metadata = []

    for path in file_paths:
        print(f"Processing file: {path}")

        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        text = ""

        if path.endswith(".pdf"):
            reader = PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() or ""

        elif path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        else:
            print(f"Unsupported file type: {path}")
            continue

        text = text.replace("\n", " ").strip()

        docs.append(text)
        metadata.append({
            "path": path,
            "hash": file_hash(path)
        })

    return docs, metadata


# -------------------------
# CHUNKING
# -------------------------
def chunk_text(text, chunk_size=250, overlap=40):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -------------------------
# EMBEDDING
# -------------------------
def load_embedder():
    print("Loading embedding model...")
    return SentenceTransformer(MODEL_NAME, device="cuda")


def generate_embeddings(model, chunks, batch_size=32):
    print("Generating embeddings...")
    embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True)
        embeddings.append(emb)

    return np.vstack(embeddings)


# -------------------------
# FAISS
# -------------------------
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# -------------------------
# LOAD EXISTING
# -------------------------
def load_existing():
    if os.path.exists(PROCESSED_PATH) and os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing index...")

        with open(PROCESSED_PATH, "rb") as f:
            data = pickle.load(f)

        index = faiss.read_index(FAISS_INDEX_PATH)

        return data.get("chunks", []), data.get("metadata", []), index

    return [], [], None


# -------------------------
# SAVE
# -------------------------
def save_all(chunks, metadata, index):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/index", exist_ok=True)

    with open(PROCESSED_PATH, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "metadata": metadata
        }, f)

    faiss.write_index(index, FAISS_INDEX_PATH)


# -------------------------
# MAIN
# -------------------------
def main():
    print("MAIN STARTED")

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file1> <file2> ...")
        return

    file_paths = sys.argv[1:]
    print("Input files:", file_paths)

    start_time = time.time()

    # Load existing
    existing_chunks, existing_meta, existing_index = load_existing()

    # Safe hash extraction
    existing_hashes = set()
    for m in existing_meta:
        if "hash" in m:
            existing_hashes.add(m["hash"])

    # Load docs
    docs, metadata = load_documents(file_paths)

    print("Docs loaded:", len(docs))

    if not docs:
        print("No valid documents loaded.")
        return

    # Filter duplicates
    new_docs = []
    new_meta = []

    for doc, meta in zip(docs, metadata):
        if meta["hash"] not in existing_hashes:
            new_docs.append(doc)
            new_meta.append(meta)
        else:
            print(f"Skipping duplicate: {meta['path']}")

    if not new_docs:
        print("No new documents.")
        return

    # Chunking
    print("Chunking...")
    all_chunks = []
    chunk_metadata = []

    for doc, meta in zip(new_docs, new_meta):
        chunks = chunk_text(doc)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                "source": meta["path"],
                "chunk_id": i
            })

    print("Total chunks:", len(all_chunks))

    # Embeddings
    embedder = load_embedder()
    embeddings = generate_embeddings(embedder, all_chunks)

    # Index
    if existing_index is not None:
        print("Merging with existing index...")
        existing_index.add(embeddings)
        final_chunks = existing_chunks + all_chunks
        final_meta = existing_meta + chunk_metadata
        index = existing_index
    else:
        print("Creating new index...")
        index = build_faiss_index(embeddings)
        final_chunks = all_chunks
        final_meta = chunk_metadata

    # Save
    print("Saving...")
    save_all(final_chunks, final_meta, index)

    print(f"Done in {time.time() - start_time:.2f}s")


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    main()