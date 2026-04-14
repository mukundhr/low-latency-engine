import time
from app.embedder import Embedder
from app.retriever import Retriever
from app.cache import SimpleCache
from app.llm import LLM

def detect_answer_type(query):
    q = query.lower()

    if any(word in q for word in ["step", "steps", "process", "flow", "pipeline"]):
        return "steps"

    elif any(word in q for word in ["component", "components", "module", "modules", "part"]):
        return "components"

    elif any(word in q for word in ["why", "reason", "purpose"]):
        return "explanation"

    else:
        return "general"
# -------------------------
# QUERY CLASSIFICATION
# -------------------------
def classify_query(query):
    length = len(query.split())

    if length <= 5:
        return "simple"
    elif length <= 12:
        return "medium"
    else:
        return "complex"


# -------------------------
# ADAPTIVE CONFIG
# -------------------------
def get_config(query):
    q_type = classify_query(query)

    if q_type == "simple":
        return {
            "top_k": 2,
            "num_predict": 80
        }

    elif q_type == "medium":
        return {
            "top_k": 3,
            "num_predict": 120
        }

    else:
        return {
            "top_k": 4,
            "num_predict": 180
        }


# -------------------------
# RERANK (semantic + keyword)
# -------------------------
def rerank(query, results):
    query_words = set(query.lower().split())

    scored = []

    for r in results:
        text_words = set(r["text"].lower().split())
        overlap = len(query_words & text_words)

        score = r["score"] + 0.1 * overlap
        scored.append((score, r))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [r for _, r in scored]


# -------------------------
# REMOVE DUPLICATES
# -------------------------
def deduplicate(results):
    seen = set()
    unique = []

    for r in results:
        key = r["text"][:60]

        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


# -------------------------
# MAIN PIPELINE
# -------------------------
class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.retriever = Retriever()
        self.cache = SimpleCache()
        self.llm = LLM()

    # -------------------------
    # PROMPT (BALANCED)
    # -------------------------
    def build_prompt(self, query, context):
        answer_type = detect_answer_type(query)

        if answer_type == "steps":
            instruction = """
Extract the exact steps from the context.

Rules:
- Copy the steps exactly as written
- Do NOT rephrase or summarize
- Do NOT add extra items
- Do NOT infer missing steps
- Preserve original numbering

Return ONLY the steps.
"""

        elif answer_type == "components":
            instruction = """
Extract the main components exactly from the context.

Rules:
- Use only phrases present in the text
- Do NOT rename or reinterpret components
- Do NOT add new components
"""

        elif answer_type == "explanation":
            instruction = """
Answer clearly using only the context.
"""

        else:
            instruction = """
Answer concisely using only relevant information from the context.
"""

        return f"""
{instruction}

Context:
{context}

Question:
{query}
"""

    # -------------------------
    # NORMAL QUERY
    # -------------------------
    def query(self, user_query):
        start = time.time()

        # Adaptive config
        config = get_config(user_query)
        top_k = config["top_k"]
        num_predict = config["num_predict"]

        # Cache
        cached = self.cache.get(user_query)
        if cached:
            return {
                "answer": cached,
                "cached": True
            }

        # -------------------------
        # EMBEDDING
        # -------------------------
        t0 = time.time()
        query_embedding = self.embedder.encode(user_query)
        embed_time = time.time() - t0

        # -------------------------
        # RETRIEVAL
        # -------------------------
        t1 = time.time()

        initial_k = max(top_k * 3, 6)
        candidates = self.retriever.search(query_embedding, initial_k)

        # -------------------------
        # RERANK + CLEAN
        # -------------------------
        results = rerank(user_query, candidates)
        results = deduplicate(results)
        results = results[:top_k]

        retrieval_time = time.time() - t1

        # -------------------------
        # CONTEXT (STRUCTURED)
        # -------------------------
        context = "\n\n".join([r["text"][:150] for r in results])

        # -------------------------
        # GENERATION
        # -------------------------
        t2 = time.time()
        prompt = self.build_prompt(user_query, context)
        answer = self.llm.generate(prompt, num_predict=num_predict)
        generation_time = time.time() - t2

        # Cache
        self.cache.set(user_query, answer)

        return {
            "answer": answer,
            "sources": results,
            "latency": time.time() - start,
            "config": config,
            "breakdown": {
                "embedding": embed_time,
                "retrieval": retrieval_time,
                "generation": generation_time
            }
        }

    # -------------------------
    # STREAMING QUERY
    # -------------------------
    def stream_query(self, user_query):
        config = get_config(user_query)
        top_k = config["top_k"]
        num_predict = config["num_predict"]

        # Embedding
        query_embedding = self.embedder.encode(user_query)

        # Retrieval
        initial_k = max(top_k * 3, 6)
        candidates = self.retriever.search(query_embedding, initial_k)

        # Rerank + clean
        results = rerank(user_query, candidates)
        results = deduplicate(results)
        results = results[:top_k]

        # Context
        context = "\n\n".join([r["text"][:150] for r in results])

        prompt = self.build_prompt(user_query, context)

        # Stream
        for token in self.llm.stream_generate(prompt, num_predict):
            yield token