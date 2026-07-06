import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from research_copilot.config import settings

VECTOR_SIZE = 384
BATCH_SIZE = 64


def prepare_documents(papers: list[dict]) -> list[dict]:
    documents = []
    for paper in papers:
        documents.append(
            {
                "page_content": f"{paper['title']}. {paper['abstract']}",
                "metadata": {
                    "openalex_id": paper["openalex_id"],
                    "title": paper["title"],
                    "authors": ", ".join(paper["authors"]),
                    "cited_by_count": paper["cited_by_count"],
                    "publication_year": paper["publication_year"],
                    "concepts": ", ".join(paper["concepts"]),
                    "doi": paper["doi"],
                    "oa_url": paper["oa_url"],
                },
            }
        )
    return documents


def embed_and_upsert(documents: list[dict]) -> None:
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    client = QdrantClient(url=settings.QDRANT_URL)

    if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    processed = 0
    last_printed = 0

    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch = documents[batch_start : batch_start + BATCH_SIZE]
        texts = [doc["page_content"] for doc in batch]
        embeddings = model.encode(texts, batch_size=BATCH_SIZE)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"page_content": doc["page_content"], **doc["metadata"]},
            )
            for doc, embedding in zip(batch, embeddings)
        ]
        client.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=points)

        processed += len(batch)
        if processed - last_printed >= 100:
            print(f"Upserted {processed} docs")
            last_printed = processed

    if processed > last_printed:
        print(f"Upserted {processed} docs")
