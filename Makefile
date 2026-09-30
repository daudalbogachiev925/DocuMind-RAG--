.PHONY: up down llm pull index api ui eval test

up:
	docker compose up -d qdrant postgres elasticsearch ollama

down:
	docker compose down -v

llm:
	docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_K_M

pull:
	docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_K_M

index:
	docker compose run --rm indexer python index.py

api:
	docker compose up -d --build api

ui:
	docker compose up -d --build ui

eval:
	docker compose exec api python /app/../eval/ragas_eval.py || python eval/ragas_eval.py

test:
	pytest -q tests/
