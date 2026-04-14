from fastapi import FastAPI
from app.rag_pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline()


@app.get("/")
def root():
    return {"message": "RAG system running"}


@app.get("/query")
def query(q: str):
    return rag.query(q)