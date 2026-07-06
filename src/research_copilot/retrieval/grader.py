import json

from research_copilot.llm.provider import extract_json, get_llm

GRADE_PROMPT = (
    "Given this query: {query}, is this document relevant? "
    'Document: {document}. Reply with JSON: {{"relevant": true/false, "reason": string}}'
)


def grade_documents(query: str, docs: list[dict]) -> list[dict]:
    llm = get_llm()
    relevant_docs = []

    for doc in docs:
        metadata = doc.get("metadata", {})
        document_text = f"{metadata.get('title', '')}. {doc.get('page_content', '')[:1500]}"
        prompt = GRADE_PROMPT.format(query=query, document=document_text)
        response = llm.invoke(prompt)

        try:
            result = extract_json(response.content)
            is_relevant = bool(result.get("relevant"))
        except (json.JSONDecodeError, ValueError, AttributeError):
            is_relevant = False

        if is_relevant:
            relevant_docs.append(doc)

    return relevant_docs
