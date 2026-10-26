"""BM25-style keyword search using SQLite's built-in FTS5 extension.
Handles exact identifiers, codes, and names that embeddings often miss."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from db import get_connection
from config import TOP_K_RETRIEVAL, EMBEDDING_VERSION


def _sanitize_fts_query(query: str) -> str:
    """FTS5 MATCH has its own query syntax; wrap each term in quotes and
    OR them together so punctuation/special characters in the user's
    query don't raise a syntax error."""
    terms = [t for t in query.replace('"', " ").split() if t]
    return " OR ".join(f'"{t}"' for t in terms) if terms else '""'


def keyword_search(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    access_levels: list = None,
    department: str = None,
    article_ref: str = None,
) -> list:
    """Returns a list of dicts: {chunk_id, text, source, page, score}."""
    conn = get_connection()
    fts_query = _sanitize_fts_query(query)

    sql = """
        SELECT c.chunk_id, c.text, c.page, c.article_ref, d.source,
               bm25(chunks_fts) AS rank_score
        FROM chunks_fts
        JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
        JOIN documents d ON d.document_id = c.document_id
        WHERE chunks_fts MATCH ? AND d.status = 'ready'
          AND c.embedding_version = ?
    """
    params = [fts_query, EMBEDDING_VERSION]

    if access_levels:
        placeholders = ",".join("?" for _ in access_levels)
        sql += f" AND c.access_level IN ({placeholders})"
        params.extend(access_levels)
    if department:
        sql += " AND c.department = ?"
        params.append(department)
    if article_ref is not None:
        sql += " AND c.article_ref = ?"
        params.append(article_ref)

    sql += " ORDER BY rank_score LIMIT ?"
    params.append(top_k)

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    # FTS5's bm25() is lower-is-better; invert so higher = more relevant,
    # matching the convention used by vector_search's cosine similarity.
    return [
        {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "source": row["source"],
            "page": row["page"],
            "article_ref": row["article_ref"],
            "score": -row["rank_score"],
        }
        for row in rows
    ]
