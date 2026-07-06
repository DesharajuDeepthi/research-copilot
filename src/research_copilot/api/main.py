from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from research_copilot.agents.graph import app as rag_app

app = FastAPI(title="Research Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str


class Citation(BaseModel):
    title: str | None
    openalex_id: str
    doi: str | None
    year: int | None
    cited_by: int | None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    agent_trace: list[str]
    fallback_used: bool
    grounding_passed: bool
    citation_validated: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    initial_state = {
        "query": request.query,
        "rewritten_query": None,
        "retrieved_docs": [],
        "graded_docs": [],
        "generation": None,
        "citations": [],
        "agent_trace": [],
        "fallback_used": False,
        "grounding_passed": False,
        "citation_validated": False,
        "retry_count": 0,
    }

    result = rag_app.invoke(initial_state, config={"recursion_limit": 25})

    doc_lookup = {
        doc.get("metadata", {}).get("openalex_id"): doc.get("metadata", {})
        for doc in result.get("graded_docs", [])
    }

    citations = [
        Citation(
            title=doc_lookup.get(citation_id, {}).get("title"),
            openalex_id=citation_id,
            doi=doc_lookup.get(citation_id, {}).get("doi"),
            year=doc_lookup.get(citation_id, {}).get("publication_year"),
            cited_by=doc_lookup.get(citation_id, {}).get("cited_by_count"),
        )
        for citation_id in result.get("citations", [])
    ]

    return AskResponse(
        answer=result.get("generation") or "",
        citations=citations,
        agent_trace=result.get("agent_trace", []),
        fallback_used=result.get("fallback_used", False),
        grounding_passed=result.get("grounding_passed", False),
        citation_validated=result.get("citation_validated", False),
    )
