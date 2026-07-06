from research_copilot.ingestion.chunker import prepare_documents
from research_copilot.ingestion.openalex import search_live
from research_copilot.llm.provider import get_llm
from research_copilot.retrieval.grader import grade_documents
from research_copilot.retrieval.hybrid import HybridRetriever
from research_copilot.retrieval.rewriter import rewrite_query as rewrite_query_llm
from research_copilot.validation.citations import validate_citations as check_citations
from research_copilot.validation.grounding import check_grounding as check_grounding_llm

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a research assistant that answers questions using only the provided context. "
    "You must cite the paper title and OpenAlex ID for every claim, in the form "
    "(Title, openalex_id). Do not use knowledge outside the provided context, and do not "
    "hallucinate facts, papers, or citations that are not present in the context."
)

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieve(state: dict) -> dict:
    query = state.get("rewritten_query") or state["query"]
    docs = _get_retriever().search(query, top_k=10)

    trace = state.get("agent_trace", []) + [f"retrieve: found {len(docs)} docs for query '{query}'"]
    return {"retrieved_docs": docs, "agent_trace": trace}


def grade_retrieval(state: dict) -> dict:
    query = state.get("rewritten_query") or state["query"]
    retrieved_docs = state["retrieved_docs"]
    graded_docs = grade_documents(query, retrieved_docs)

    trace = state.get("agent_trace", []) + [
        f"Graded {len(retrieved_docs)} docs, {len(graded_docs)} relevant"
    ]
    return {"graded_docs": graded_docs, "agent_trace": trace}


def rewrite_query(state: dict) -> dict:
    original = state["query"]
    new_query = rewrite_query_llm(original)

    trace = state.get("agent_trace", []) + [f"Query rewritten: {original} → {new_query}"]
    return {
        "rewritten_query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "agent_trace": trace,
    }


def live_api_fallback(state: dict) -> dict:
    query = state.get("rewritten_query") or state["query"]
    papers = search_live(query, per_page=5)
    new_docs = prepare_documents(papers)

    retrieved_docs = state.get("retrieved_docs", []) + new_docs
    graded_docs = state.get("graded_docs", []) + new_docs

    trace = state.get("agent_trace", []) + [
        f"Live API fallback triggered — fetched {len(new_docs)} fresh papers from OpenAlex"
    ]
    return {
        "retrieved_docs": retrieved_docs,
        "graded_docs": graded_docs,
        "fallback_used": True,
        "agent_trace": trace,
    }


def synthesize(state: dict) -> dict:
    docs = state["graded_docs"]
    context = "\n\n".join(
        f"[{i}] Title: {doc.get('metadata', {}).get('title')}\n"
        f"OpenAlex ID: {doc.get('metadata', {}).get('openalex_id')}\n"
        f"Content: {doc.get('page_content', '')[:1500]}"
        for i, doc in enumerate(docs, start=1)
    )
    human_prompt = f"Question: {state['query']}\n\nContext:\n{context}\n\nAnswer:"

    llm = get_llm()
    generation = llm.invoke(
        [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": human_prompt},
        ]
    ).content.strip()

    known_ids = {doc.get("metadata", {}).get("openalex_id") for doc in docs}
    citations = [doc_id for doc_id in known_ids if doc_id and doc_id in generation]

    trace = state.get("agent_trace", []) + [
        f"synthesize: generated answer citing {len(citations)} OpenAlex IDs"
    ]
    return {"generation": generation, "citations": citations, "agent_trace": trace}


def check_grounding(state: dict) -> dict:
    result = check_grounding_llm(state["generation"], state["graded_docs"])
    grounded = result["grounded"]
    unsupported_claims = result["unsupported_claims"]

    if grounded:
        trace_line = "check_grounding: PASSED"
    else:
        trace_line = f"check_grounding: FAILED - unsupported claims: {unsupported_claims}"

    trace = state.get("agent_trace", []) + [trace_line]
    return {"grounding_passed": grounded, "agent_trace": trace}


def validate_citations(state: dict) -> dict:
    citations = state["citations"]
    passed, invalid = check_citations(citations, state["graded_docs"])

    if passed:
        trace_line = f"validate_citations: PASS - all {len(citations)} citations verified"
    else:
        trace_line = f"validate_citations: FAIL - invalid citations: {invalid}"

    trace = state.get("agent_trace", []) + [trace_line]
    return {"citation_validated": passed, "agent_trace": trace}
