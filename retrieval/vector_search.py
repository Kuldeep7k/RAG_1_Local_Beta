"""Vector similarity search against Qdrant, with optional permission
filtering applied at the database layer (never just in the prompt)."""
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from config import QDRANT_COLLECTION, TOP_K_RETRIEVAL, EMBEDDING_DIM, EMBEDDING_VERSION, INDEX_SCHEMA_VERSION
from embeddings import embed_text
from qdrant_utils import get_qdrant_client


def build_qdrant_filter(access_levels: list = None, department: str = None, article_ref: str = None):
    conditions = []
    if access_levels:
        conditions.append(FieldCondition(key="access_level", match=MatchAny(any=access_levels)))
    if department:
        conditions.append(FieldCondition(key="department", match=MatchValue(value=department)))
    if article_ref is not None:
        conditions.append(FieldCondition(key="article_ref", match=MatchValue(value=article_ref)))
    conditions.append(FieldCondition(key="embedding_version", match=MatchValue(value=EMBEDDING_VERSION)))
    conditions.append(FieldCondition(key="index_schema_version", match=MatchValue(value=INDEX_SCHEMA_VERSION)))
    return Filter(must=conditions) if conditions else None


def vector_search(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    access_levels: list = None,
    department: str = None,
    article_ref: str = None,
) -> list:
    """Returns a list of dicts: {chunk_id, text, source, page, score}."""
    client = get_qdrant_client()
    query_vector = embed_text(query)
    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(f"Query embedding dimension {len(query_vector)} does not match configured dimension {EMBEDDING_DIM}")
    query_filter = build_qdrant_filter(access_levels, department, article_ref)

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    return [
        {
            "chunk_id": hit.payload["chunk_id"],
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "page": hit.payload["page"],
            "article_number": hit.payload.get("article_number"),
            "article_ref": hit.payload.get("article_ref"),
            "score": hit.score,
        }
        for hit in response.points
    ]
