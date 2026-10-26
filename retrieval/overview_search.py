"""Representative retrieval for document-level overview questions."""
from db import get_connection
from config import TOP_K_RETRIEVAL, EMBEDDING_VERSION


def document_overview_search(
    top_k: int = TOP_K_RETRIEVAL,
    access_levels: list = None,
    department: str = None,
) -> list:
    """Return representative chunks from each accessible ready document."""
    conn = get_connection()
    conditions = ["d.status = 'ready'", "c.embedding_version = ?"]
    params = [EMBEDDING_VERSION]
    if access_levels:
        placeholders = ",".join("?" for _ in access_levels)
        conditions.append(f"c.access_level IN ({placeholders})")
        params.extend(access_levels)
    if department:
        conditions.append("c.department = ?")
        params.append(department)

    where = " AND ".join(conditions)
    rows = conn.execute(
        f"""
        WITH ranked AS (
            SELECT c.chunk_id, c.text, c.page, c.article_ref, d.source,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.document_id
                       ORDER BY c.page, c.chunk_id
                   ) AS document_rank
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE {where}
        )
        SELECT chunk_id, text, page, article_ref, source
        FROM ranked
        WHERE document_rank <= ?
        ORDER BY source, document_rank
        LIMIT ?
        """,
        (*params, max(1, min(8, top_k)), top_k),
    ).fetchall()
    conn.close()
    return [
        {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "source": row["source"],
            "page": row["page"],
            "article_ref": row["article_ref"],
            "score": 0.0,
        }
        for row in rows
    ]
