def validate_citations(citations: list[str], docs: list[dict]) -> tuple[bool, list[str]]:
    known_ids = {doc.get("metadata", {}).get("openalex_id") for doc in docs}
    invalid = [citation for citation in citations if citation not in known_ids]
    return not invalid, invalid
