FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    fastapi==0.111.0 uvicorn[standard]==0.30.1 \
    sentence-transformers==2.7.0 \
    qdrant-client==1.9.1 elasticsearch==8.13.0 \
    rank-bm25==0.2.2 requests==2.32.0 \
    prometheus-client==0.20.0
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
