FROM python:3.11-slim
RUN pip install --no-cache-dir streamlit==1.35.0 requests==2.32.0
WORKDIR /app
COPY app.py .
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
