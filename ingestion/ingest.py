"""Ties parsing -> chunking -> embedding -> storage (Qdrant + SQLite FTS5)
into a single ingest_file() call. Can also be run directly from the CLI."""
import sys
import uuid
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue,
)

from config import (
    QDRANT_COLLECTION, EMBEDDING_DIM, EMBEDDING_VERSION,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, DEFAULT_ACCESS_LEVEL, INDEX_SCHEMA_VERSION,
)
from db import get_connection, init_db, update_job_progress, is_job_cancelled
from embeddings import embed_batch
from ingestion.loaders import load_document
from ingestion.chunker import chunk_document
from config import EMBED_BATCH_SIZE
from qdrant_utils import get_qdrant_client

logger = logging.getLogger(__name__)


class IngestionCancelled(Exception):
    """Raised when an active ingestion job is cancelled by its owner."""


def _check_cancelled(job_id: int | None):
    if job_id is not None and is_job_cancelled(job_id):
        raise IngestionCancelled("Ingestion was cancelled")


def cleanup_document(document_id: str, client: QdrantClient = None):
    """Remove a newly-created document from both stores after cancellation."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        conn.commit()
    finally:
        conn.close()
    try:
        (client or get_qdrant_client()).delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]),
        )
    except Exception:
        logger.warning("Failed to remove partial vectors for document %s", document_id, exc_info=True)


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        return

    collection = client.get_collection(QDRANT_COLLECTION)
    configured_size = collection.config.params.vectors.size
    if configured_size != EMBEDDING_DIM:
        raise ValueError(
            f"Qdrant collection {QDRANT_COLLECTION!r} expects vectors of size "
            f"{configured_size}, but the configured embedding dimension is {EMBEDDING_DIM}. "
            "Use a new collection or re-index with the matching embedding model."
        )


def replace_existing_documents(source_name: str, old_document_ids: list[str], client: QdrantClient):
    """Remove superseded documents after their replacement is ready.

    SQLite and Qdrant do not share a transaction. The replacement is therefore
    indexed first and exposed as ready before old records are removed, which
    preserves searchability if parsing, embedding, or vector upsert fails.
    """
    if not old_document_ids:
        return
    # Remove old vectors first. If this external operation fails, the old
    # SQLite rows remain available to keyword search instead of leaving stale
    # vectors with no matching metadata behind.
    for document_id in old_document_ids:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]),
        )

    conn = get_connection()
    try:
        for document_id in old_document_ids:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        conn.commit()
    finally:
        conn.close()

    print(f"Replaced {len(old_document_ids)} prior ingestion(s) of '{source_name}'.")


def ingest_file(
    file_path: str,
    access_level: str = DEFAULT_ACCESS_LEVEL,
    department: str = None,
    job_id: int = None,
    source_name: str = None,
) -> str:
    """Ingests a single file end-to-end. Returns the new document_id.
    If job_id is given, writes live progress (total_chunks/chunks_done)
    to the jobs table so the API/UI can display it."""
    init_db()
    _check_cancelled(job_id)
    client = get_qdrant_client()
    ensure_collection(client)

    source_name = source_name or Path(file_path).name
    conn = get_connection()
    try:
        old_document_ids = [
            row["document_id"]
            for row in conn.execute(
                "SELECT document_id FROM documents WHERE source = ? AND status = 'ready'", (source_name,)
            ).fetchall()
        ]
    finally:
        conn.close()

    document_id = str(uuid.uuid4())

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO documents (document_id, source, access_level, department, status) VALUES (?, ?, ?, ?, 'indexing')",
            (document_id, source_name, access_level, department),
        )
        conn.commit()  # small, fast transaction -- don't hold the write lock past this

        pages = load_document(file_path)
        chunks = chunk_document(pages, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
        total_chunks = len(chunks)
        print(f"Parsed {source_name}: {len(pages)} pages -> {total_chunks} chunks. Embedding...")

        if job_id is not None:
            update_job_progress(job_id, total_chunks=total_chunks, chunks_done=0)

        points = []
        chunks_done = 0
        for batch_start in range(0, total_chunks, EMBED_BATCH_SIZE):
            _check_cancelled(job_id)
            batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
            vectors = embed_batch(
                [c["text"] for c in batch],
                on_progress=(
                    lambda completed: update_job_progress(
                        job_id, chunks_done=batch_start + completed
                    )
                ) if job_id is not None else None,
            )
            if any(len(vector) != EMBEDDING_DIM for vector in vectors):
                raise ValueError(
                    f"Embedding provider returned a vector with unexpected dimension; "
                    f"expected {EMBEDDING_DIM}."
                )

            for chunk, vector in zip(batch, vectors):
                chunk_id = f"{document_id}_{chunk['page']}_{chunk['chunk_index']}"

                conn.execute(
                    """INSERT INTO chunks
                       (chunk_id, document_id, text, page, section, article_number, article_ref, access_level, department, embedding_version)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (chunk_id, document_id, chunk["text"], chunk["page"], None, chunk.get("article_number"),
                     chunk.get("article_ref"), access_level, department, EMBEDDING_VERSION),
                )

                points.append(PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "source": source_name,
                        "page": chunk["page"],
                        "text": chunk["text"],
                        "article_number": chunk.get("article_number"),
                        "article_ref": chunk.get("article_ref"),
                        "access_level": access_level,
                        "department": department,
                        "embedding_version": EMBEDDING_VERSION,
                        "index_schema_version": INDEX_SCHEMA_VERSION,
                    },
                ))

            # Commit and report progress once per batch rather than once per
            # chunk -- keeps the write lock brief and avoids hundreds of
            # tiny DB writes.
            chunks_done = batch_start + len(batch)
            conn.commit()
            print(f"  embedded {chunks_done}/{total_chunks} chunks")
            if job_id is not None:
                update_job_progress(job_id, chunks_done=chunks_done)

        _check_cancelled(job_id)

        if not points:
            raise ValueError("No chunks were produced from the document.")

        if points:
            _check_cancelled(job_id)
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)

        _check_cancelled(job_id)

        # Only now is the replacement visible to keyword retrieval. Vector
        # points are also uploaded only after every batch has embedded.
        conn.execute("UPDATE documents SET status = 'ready' WHERE document_id = ?", (document_id,))
        conn.commit()
        _check_cancelled(job_id)
        replace_existing_documents(source_name, old_document_ids, client)

        print(f"Ingested {total_chunks} chunks from {source_name} (document_id={document_id})")
        return document_id
    except Exception:
        # Preserve the previously ready document, but remove any partial
        # replacement from both stores so it cannot surface in later search.
        try:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
        finally:
            try:
                client.delete(
                    collection_name=QDRANT_COLLECTION,
                    points_selector=Filter(must=[
                        FieldCondition(key="document_id", match=MatchValue(value=document_id))
                    ]),
                )
            except Exception:
                # Do not mask the original ingestion error. A later manual
                # re-upload of the same source will replace residual points.
                logger.warning(
                    "Failed to remove partial vectors for document %s",
                    document_id,
                    exc_info=True,
                )
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a document into the RAG system.")
    parser.add_argument("file_path", help="Path to a PDF, DOCX, or TXT/MD file")
    parser.add_argument("--access-level", default=DEFAULT_ACCESS_LEVEL)
    parser.add_argument("--department", default=None)
    args = parser.parse_args()

    ingest_file(args.file_path, args.access_level, args.department)
