# Research Copilot

A Corrective RAG (Retrieval-Augmented Generation) research assistant over academic papers from
[OpenAlex](https://openalex.org/). It combines hybrid search (BM25 + vector), an LLM-driven
self-correcting agent loop (grade → rewrite/fallback → synthesize → verify), and a Streamlit chat
UI that shows the agent's reasoning trace in real time.

Ask a research question, and the system:

1. Retrieves candidate papers with hybrid search.
2. Grades each one for relevance and drops the irrelevant ones.
3. If too few relevant docs are found, it rewrites the query and retries — and if that still
   fails, it falls back to a live OpenAlex search.
4. Synthesizes an answer that cites paper titles and OpenAlex IDs.
5. Checks the answer is grounded in the retrieved sources (no hallucinated claims).
6. Deterministically verifies every citation actually exists in the retrieved set.

See [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) for the full architecture writeup.

## Project structure

```
research-copilot/
├── src/research_copilot/
│   ├── config.py            # pydantic-settings config (env vars)
│   ├── ingestion/
│   │   ├── openalex.py      # OpenAlex API client (paginated fetch + live search)
│   │   └── chunker.py       # doc prep + embedding + Qdrant upsert
│   ├── retrieval/
│   │   ├── hybrid.py        # BM25 + vector search, Reciprocal Rank Fusion
│   │   ├── grader.py        # LLM relevance grading
│   │   └── rewriter.py      # LLM query rewriting
│   ├── agents/
│   │   ├── graph.py         # LangGraph StateGraph (the Corrective RAG pipeline)
│   │   └── nodes.py         # node implementations
│   ├── validation/
│   │   ├── grounding.py     # LLM groundedness check
│   │   └── citations.py     # deterministic citation verification
│   ├── llm/provider.py      # LLM client factory
│   └── api/main.py          # FastAPI app (POST /ask, GET /health)
├── ui/app.py                # Streamlit chat UI
├── eval/                    # evaluation harness (scaffolded, not yet implemented)
├── scripts/ingest.py        # CLI ingestion entrypoint (scaffolded, not yet implemented)
├── docker-compose.yml       # api + qdrant services
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Prerequisites

- Python 3.10+
- Docker (for Qdrant, and optionally for running the API in a container)
- An LLM API key — the project currently runs on OpenAI (`gpt-4o-mini` by default)
- An OpenAlex-polite-pool email (just your own email address, no signup required)

## Getting started

1. **Install dependencies**

   ```bash
   make install
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Fill in `.env`:

   | Variable | Description |
   |---|---|
   | `OPENAI_API_KEY` | Your OpenAI API key |
   | `OPENAI_MODEL` | Defaults to `gpt-4o-mini` |
   | `ANTHROPIC_API_KEY` | Optional — only needed if you switch `llm/provider.py` back to Claude |
   | `QDRANT_URL` | `http://localhost:6333` for local Docker Qdrant |
   | `QDRANT_COLLECTION_NAME` | Any name, e.g. `research_papers` |
   | `OPENALEX_EMAIL` | Your email, sent as a `User-Agent` header to OpenAlex |

3. **Start Qdrant**

   ```bash
   docker compose up -d qdrant
   ```

4. **Ingest some papers**

   Currently done via a short Python snippet (a dedicated `scripts/ingest.py` CLI is scaffolded
   but not yet implemented):

   ```python
   from research_copilot.ingestion.openalex import fetch_papers
   from research_copilot.ingestion.chunker import prepare_documents, embed_and_upsert
   from research_copilot.retrieval.hybrid import HybridRetriever

   papers = fetch_papers(["data engineering"], max_per_topic=50)
   docs = prepare_documents(papers)
   embed_and_upsert(docs)

   retriever = HybridRetriever(index_path="data/bm25_index.json")
   retriever.build_bm25(docs)
   ```

5. **Run the API**

   ```bash
   make run-api
   ```

6. **Run the UI**

   ```bash
   make run-ui
   ```

   Open http://localhost:8501 and ask a question.

### Running with Docker

```bash
docker compose up -d qdrant
docker build -t research-copilot .
docker run --rm --network research-copilot_default \
  --env-file .env -e QDRANT_URL=http://research-copilot-qdrant-1:6333 \
  -p 8000:8000 research-copilot
```

## API

### `POST /ask`

Request:
```json
{ "query": "What is data engineering?" }
```

Response:
```json
{
  "answer": "Data engineering involves ... (Title, https://openalex.org/W123)",
  "citations": [
    { "title": "...", "openalex_id": "https://openalex.org/W123", "doi": "...", "year": 2004, "cited_by": 167 }
  ],
  "agent_trace": [
    "retrieve: found 10 docs for query '...'",
    "Graded 10 docs, 6 relevant",
    "synthesize: generated answer citing 2 OpenAlex IDs",
    "check_grounding: PASSED",
    "validate_citations: PASS - all 2 citations verified"
  ],
  "fallback_used": false,
  "grounding_passed": true,
  "citation_validated": true
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Makefile targets

| Target | Description |
|---|---|
| `make install` | Install the package and dev dependencies |
| `make ingest` | Run `scripts/ingest.py` (not yet implemented) |
| `make run-api` | Start the FastAPI backend with `uvicorn` |
| `make run-ui` | Start the Streamlit UI |
| `make eval` | Run `eval/run_eval.py` (not yet implemented) |
| `make test` | Run `pytest` |
| `make lint` | Run `ruff check` |

## Known gaps / roadmap

- `scripts/ingest.py` — CLI wrapper around the ingestion pipeline shown above
- `eval/` — RAGAS-based evaluation harness against `eval/golden_set.jsonl`
- `tests/` — unit tests
- `agents/tools.py` — scaffolded, unused
- `retrieval/grader.py` grades documents one LLM call at a time; batching would reduce latency/cost
