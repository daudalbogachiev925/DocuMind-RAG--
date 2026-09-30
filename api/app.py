"""Гибридный поиск + RAG-генерация (Ollama) с цитатами."""
import os, json, logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

log = logging.getLogger("api"); logging.basicConfig(level=logging.INFO)

QDRANT = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
ES = Elasticsearch(os.getenv("ELASTIC_URL", "http://elasticsearch:9200"))
OLLAMA = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMB = SentenceTransformer("BAAI/bge-small-en-v1.5")
COLLECTION = "docs"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")

REQ = Counter("dm_requests_total", "Requests", ["endpoint", "status"])
LAT = Histogram("dm_latency_seconds", "Latency", ["endpoint"])

app = FastAPI(title="DocuMind API", version="1.0.0")

class Query(BaseModel):
    q: str
    top_k: int = 20
    rerank: int = 5
    generate: bool = True

def vector_search(q, k):
    v = EMB.encode([q], normalize_embeddings=True)[0].tolist()
    hits = QDRANT.search(COLLECTION, query_vector=v, limit=k)
    return [{"id": h.id, "score": h.score, **h.payload} for h in hits]

def bm25_search(q, k):
    res = ES.search(index="docs", size=k, query={
        "multi_match": {"query": q, "fields": ["text^2"]}
    })
    return [{"id": h["_id"], "score": h["_score"], **h["_source"]} for h in res["hits"]["hits"]]

def rrf(rankings: List[List[dict]], k=60):
    """Reciprocal Rank Fusion."""
    scores = {}
    docs = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking):
            scores[d["id"]] = scores.get(d["id"], 0) + 1 / (k + rank + 1)
            docs[d["id"]] = d
    return [docs[i] for i in sorted(scores, key=scores.get, reverse=True)]

def rerank_cross(q, docs, top):
    """Простой cross-encoder реранкинг."""
    from sentence_transformers import CrossEncoder
    global _CE
    if "_CE" not in globals():
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(q, d["text"][:512]) for d in docs]
    scores = _CE.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])[:top]
    return [{"rerank_score": float(s), **d} for d, s in ranked]

def generate(prompt, context):
    body = {
        "model": LLM_MODEL,
        "prompt": f"Контекст:\n{context}\n\nВопрос: {prompt}\n\nОтвечай только по контексту и цитируй источник [file:idx].",
        "stream": False,
    }
    r = requests.post(f"{OLLAMA}/api/generate", json=body, timeout=120)
    r.raise_for_status()
    return r.json()["response"]

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/metrics")
def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/search")
def search(q: Query):
    with LAT.labels("search").time():
        try:
            vs = vector_search(q.q, q.top_k)
            bs = bm25_search(q.q, q.top_k)
            fused = rrf([vs, bs])
            reranked = rerank_cross(q.q, fused, q.rerank)

            answer = None
            if q.generate:
                ctx = "\n\n".join(f"[{d['file']}:{d['chunk_idx']}] {d['text']}" for d in reranked)
                answer = generate(q.q, ctx)

            REQ.labels("search", "200").inc()
            return {
                "query": q.q,
                "answer": answer,
                "hits": [
                    {"file": d["file"], "chunk_idx": d["chunk_idx"],
                     "score": d["rerank_score"], "text": d["text"][:500]}
                    for d in reranked
                ],
            }
        except Exception as e:
            REQ.labels("search", "500").inc()
            raise HTTPException(500, str(e))
