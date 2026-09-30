"""Парсинг PDF/DOCX → чанки → эмбеддинги → Qdrant + Elasticsearch."""
import os, hashlib, uuid
from pathlib import Path
import pdfplumber
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from elasticsearch import Elasticsearch
import tiktoken

QDRANT = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
ES = Elasticsearch(os.getenv("ELASTIC_URL", "http://elasticsearch:9200"))
EMB = SentenceTransformer("BAAI/bge-small-en-v1.5")
TOK = tiktoken.get_encoding("cl100k_base")
COLLECTION = "docs"

def ensure():
    if not QDRANT.collection_exists(COLLECTION):
        QDRANT.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

def read_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)

def read_docx(path):
    d = DocxDocument(path)
    return "\n".join(p.text for p in d.paragraphs)

def read_txt(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")

READERS = {".pdf": read_pdf, ".docx": read_docx, ".txt": read_txt}

def chunk(text, size=400, overlap=80):
    ids = TOK.encode(text)
    out = []
    i = 0
    while i < len(ids):
        piece = ids[i:i + size]
        out.append(TOK.decode(piece))
        i += size - overlap
    return [c.strip() for c in out if len(c.strip()) > 30]

def file_id(path): return hashlib.sha256(str(path).encode()).hexdigest()[:16]

def ingest_file(path):
    ext = Path(path).suffix.lower()
    if ext not in READERS: return 0
    text = READERS[ext](path)
    if not text.strip(): return 0
    fid = file_id(path)
    chunks = chunk(text)
    if not chunks: return 0

    vectors = EMB.encode(chunks, normalize_embeddings=True, batch_size=32).tolist()
    points = []
    for i, (c, v) in enumerate(zip(chunks, vectors)):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{fid}-{i}"))
        points.append(PointStruct(
            id=pid, vector=v,
            payload={"file": str(path), "file_id": fid, "chunk_idx": i, "text": c},
        ))
        ES.index(index="docs", id=pid, document={
            "file": str(path), "file_id": fid, "chunk_idx": i, "text": c,
        })
    QDRANT.upsert(COLLECTION, points=points)
    return len(chunks)

if __name__ == "__main__":
    ensure()
    root = Path("/data")
    total = 0
    for p in root.rglob("*"):
        if p.suffix.lower() in READERS:
            n = ingest_file(p)
            print(f"✅ {p} → {n} chunks")
            total += n
    print(f"Total: {total} chunks")
