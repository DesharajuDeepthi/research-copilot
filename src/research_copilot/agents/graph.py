from typing import TypedDict

from langgraph.graph import END, StateGraph

from research_copilot.agents.nodes import (
    check_grounding,
    grade_retrieval,
    live_api_fallback,
    retrieve,
    rewrite_query,
    synthesize,
    validate_citations,
)


class State(TypedDict):
    query: str
    rewritten_query: str | None
    retrieved_docs: list[dict]
    graded_docs: list[dict]
    generation: str | None
    citations: list[str]
    agent_trace: list[str]
    fallback_used: bool
    grounding_passed: bool
    citation_validated: bool
    retry_count: int


def route_after_grading(state: State) -> str:
    if len(state["graded_docs"]) >= 2:
        return "synthesize"
    if state["retry_count"] == 0:
        return "rewrite_query"
    return "live_api_fallback"


def route_after_grounding(state: State) -> str:
    if state["grounding_passed"]:
        return "validate_citations"

    grounding_failures = sum(
        1 for entry in state["agent_trace"] if entry.startswith("check_grounding: FAILED")
    )
    if grounding_failures >= 2:
        return "validate_citations"
    return "synthesize"


graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("grade_retrieval", grade_retrieval)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("live_api_fallback", live_api_fallback)
graph.add_node("synthesize", synthesize)
graph.add_node("check_grounding", check_grounding)
graph.add_node("validate_citations", validate_citations)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "grade_retrieval")
graph.add_conditional_edges(
    "grade_retrieval",
    route_after_grading,
    {
        "rewrite_query": "rewrite_query",
        "live_api_fallback": "live_api_fallback",
        "synthesize": "synthesize",
    },
)
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("live_api_fallback", "synthesize")
graph.add_edge("synthesize", "check_grounding")
graph.add_conditional_edges(
    "check_grounding",
    route_after_grounding,
    {
        "synthesize": "synthesize",
        "validate_citations": "validate_citations",
    },
)
graph.add_edge("validate_citations", END)

app = graph.compile()
