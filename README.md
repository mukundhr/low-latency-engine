# low-latency-engine

Fast, minimal RAG service built with FastAPI, FAISS, sentence-transformers, and Ollama.

## Features
- Adaptive retrieval and generation settings based on query length
- Reranking (semantic + keyword overlap) and duplicate filtering
- Simple in-memory cache for repeat queries
- Latency breakdown in responses

## Project layout
- app/main.py: FastAPI entrypoint
- app/rag_pipeline.py: retrieval + generation pipeline
- app/embedder.py: sentence-transformers embeddings
- app/retriever.py: FAISS index lookup
- app/llm.py: Ollama client
- scripts/ingest.py: build or update FAISS index from files
- data/index: FAISS index storage
- data/processed: chunk metadata storage

## Requirements
- Python 3
- Ollama running locally for generation
- Optional: CUDA-capable GPU for embeddings

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration (optional)
Environment variables:
- EMBEDDING_MODEL (default: BAAI/bge-small-en)
- LLM_MODEL (default: mistral)
- OLLAMA_URL (default: http://localhost:11434/api/generate)

Example:
```bash
setx EMBEDDING_MODEL "BAAI/bge-small-en"
setx LLM_MODEL "llama3.2:3b"
setx OLLAMA_URL "http://localhost:11434/api/generate"
```

## Ingest documents
Build or update the FAISS index from .pdf or .txt files:
```bash
python scripts/ingest.py path\to\file1.pdf path\to\file2.txt
```

This creates:
- data/index/faiss.index
- data/processed/data.pkl

## Run the API
Terminal 1 (start Ollama):
```bash
ollama run mistral
```

Terminal 2 (start FastAPI):
```bash
uvicorn app.main:app --reload
```

Endpoints:
- GET / -> health check
- GET /query?q=... -> RAG response

Example query:
```bash
curl "http://127.0.0.1:8000/query?q=What%20is%20this%20system%3F"
```