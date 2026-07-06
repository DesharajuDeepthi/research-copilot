.PHONY: install ingest run-api run-ui eval test lint

install:
	pip install -e ".[dev]"

ingest:
	python scripts/ingest.py

run-api:
	uvicorn research_copilot.api.main:app --reload

run-ui:
	streamlit run ui/app.py

eval:
	python eval/run_eval.py

test:
	pytest

lint:
	ruff check .
