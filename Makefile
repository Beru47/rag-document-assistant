.PHONY: install lint test eval run

install:
	pip install -e ".[dev,minilm,ocr]"

lint:
	ruff check . && ruff format --check .

test:
	pytest -q

# Deterministic: hashing embedder + extractive stand-in LLM, no downloads, no API keys.
# For the real embedding model: python -m rag_assistant.eval --embeddings minilm
eval:
	python -m rag_assistant.eval --embeddings hashing --llm fake

run:
	uvicorn rag_assistant.api:app --reload
