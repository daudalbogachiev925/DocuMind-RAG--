FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential poppler-utils tesseract-ocr && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    sentence-transformers==2.7.0 \
    qdrant-client==1.9.1 \
    elasticsearch==8.13.0 \
    psycopg[binary]==3.1.18 \
    pdfplumber==0.11.0 python-docx==1.1.2 \
    rank-bm25==0.2.2 requests==2.32.0 \
    fastapi==0.111.0 uvicorn==0.30.1 \
    tiktoken==0.7.0
COPY . .
CMD ["python", "index.py", "--watch"]
