import httpx

from research_copilot.config import settings

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SELECT_FIELDS = ",".join(
    [
        "id",
        "title",
        "abstract_inverted_index",
        "authorships",
        "cited_by_count",
        "publication_year",
        "concepts",
        "doi",
        "open_access",
    ]
)


def _reconstruct_abstract(abstract_inverted_index: dict) -> str:
    positions: dict[int, str] = {}
    for word, indices in abstract_inverted_index.items():
        for index in indices:
            positions[index] = word

    return " ".join(positions[i] for i in sorted(positions))


def _parse_work(work: dict) -> dict:
    open_access = work.get("open_access") or {}
    return {
        "openalex_id": work.get("id"),
        "title": work.get("title"),
        "abstract": _reconstruct_abstract(work["abstract_inverted_index"]),
        "authors": [
            authorship.get("author", {}).get("display_name")
            for authorship in work.get("authorships", [])
        ],
        "cited_by_count": work.get("cited_by_count"),
        "publication_year": work.get("publication_year"),
        "concepts": [concept.get("display_name") for concept in work.get("concepts", [])],
        "doi": work.get("doi"),
        "oa_url": open_access.get("oa_url"),
    }


def search_live(query: str, per_page: int = 5) -> list[dict]:
    headers = {"User-Agent": f"research-copilot/1.0 (mailto:{settings.OPENALEX_EMAIL})"}
    params = {"search": query, "per_page": per_page}

    with httpx.Client(timeout=10) as client:
        response = client.get(OPENALEX_WORKS_URL, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

    return [
        _parse_work(work)
        for work in data.get("results", [])
        if work.get("abstract_inverted_index")
    ]


def fetch_papers(topics: list[str], max_per_topic: int = 500) -> list[dict]:
    headers = {"User-Agent": f"research-copilot/1.0 (mailto:{settings.OPENALEX_EMAIL})"}
    papers: list[dict] = []

    with httpx.Client(timeout=10) as client:
        for topic in topics:
            cursor = "*"
            fetched = 0

            while cursor and fetched < max_per_topic:
                params = {
                    "filter": f"title_and_abstract.search:{topic}",
                    "select": SELECT_FIELDS,
                    "cursor": cursor,
                    "per-page": min(200, max_per_topic - fetched),
                }
                response = client.get(OPENALEX_WORKS_URL, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

                for work in data.get("results", []):
                    abstract_inverted_index = work.get("abstract_inverted_index")
                    if not abstract_inverted_index:
                        continue

                    papers.append(_parse_work(work))
                    fetched += 1

                    if fetched >= max_per_topic:
                        break

                cursor = data.get("meta", {}).get("next_cursor")

            print(f"Fetched {fetched} papers for topic: {topic}")

    return papers
