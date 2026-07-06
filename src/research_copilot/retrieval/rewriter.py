from research_copilot.llm.provider import get_llm

REWRITE_PROMPT = (
    "Rewrite this academic search query to find more relevant research papers. "
    "Original: {query}. Return only the rewritten query, nothing else."
)


def rewrite_query(query: str) -> str:
    llm = get_llm()
    response = llm.invoke(REWRITE_PROMPT.format(query=query))
    return response.content.strip()
