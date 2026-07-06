import json
from pathlib import Path

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from research_copilot.config import settings

DEFAULT_BM25_INDEX_PATH = Path("data/bm25_index.json")
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return text.split()


class HybridRetriever:
    def __init__(self, index_path: str | Path = DEFAULT_BM25_INDEX_PATH):
        self.index_path = Path(index_path)
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = QdrantClient(url=settings.QDRANT_URL)

        self.documents: list[dict] = []
        self.corpus: list[str] = []
        self.bm25: BM25Okapi | None = None

        if self.index_path.exists():
            self._load_bm25()

    def build_bm25(self, documents: list[dict]) -> None:
        self.documents = documents
        self.corpus = [doc["page_content"] for doc in documents]
        tokenized_corpus = [_tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._save_bm25()

    def _save_bm25(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w") as f:
            json.dump({"documents": self.documents, "corpus": self.corpus}, f)

    def _load_bm25(self) -> None:
        with open(self.index_path) as f:
            data = json.load(f)
        self.documents = data["documents"]
        self.corpus = data["corpus"]
        tokenized_corpus = [_tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[dict, int]]:
        if self.bm25 is None or not self.corpus:
            return []

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.documents[idx], rank) for rank, idx in enumerate(ranked_indices, start=1)]

    def _vector_search(self, query: str, top_k: int) -> list[tuple[dict, int]]:
        query_vector = self.model.encode(query).tolist()
        response = self.client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        )

        results = []
        for rank, hit in enumerate(response.points, start=1):
            payload = dict(hit.payload or {})
            page_content = payload.pop("page_content", "")
            doc = {"page_content": page_content, "metadata": payload}
            results.append((doc, rank))
        return results

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        merged: dict[str, dict] = {}

        for doc, rank in self._bm25_search(query, top_k):
            key = doc["metadata"]["openalex_id"]
            entry = merged.setdefault(
                key,
                {
                    "page_content": doc["page_content"],
                    "metadata": doc["metadata"],
                    "rrf_score": 0.0,
                    "sources": set(),
                },
            )
            entry["rrf_score"] += 1 / (RRF_K + rank)
            entry["sources"].add("bm25")

        for doc, rank in self._vector_search(query, top_k):
            key = doc["metadata"]["openalex_id"]
            entry = merged.setdefault(
                key,
                {
                    "page_content": doc["page_content"],
                    "metadata": doc["metadata"],
                    "rrf_score": 0.0,
                    "sources": set(),
                },
            )
            entry["rrf_score"] += 1 / (RRF_K + rank)
            entry["sources"].add("vector")

        results = sorted(merged.values(), key=lambda e: e["rrf_score"], reverse=True)[:top_k]
        for result in results:
            result["sources"] = sorted(result["sources"])
        return results
