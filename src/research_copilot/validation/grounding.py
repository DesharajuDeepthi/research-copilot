import json

from research_copilot.llm.provider import extract_json, get_llm

GROUNDING_PROMPT = (
    "Does this answer only contain claims supported by the provided context?\n\n"
    "Answer:\n{generation}\n\nContext:\n{context}\n\n"
    'Answer JSON: {{"grounded": true/false, "unsupported_claims": list[str]}}'
)


def check_grounding(generation: str, docs: list[dict]) -> dict:
    context = "\n\n".join(
        f"[{doc.get('metadata', {}).get('openalex_id')}] {doc.get('page_content', '')[:1000]}"
        for doc in docs
    )
    llm = get_llm()
    response = llm.invoke(GROUNDING_PROMPT.format(generation=generation, context=context))

    try:
        result = extract_json(response.content)
        return {
            "grounded": bool(result.get("grounded")),
            "unsupported_claims": result.get("unsupported_claims") or [],
        }
    except (json.JSONDecodeError, ValueError, AttributeError):
        return {"grounded": False, "unsupported_claims": ["failed to parse grounding response"]}
