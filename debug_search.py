"""Debug utility: search the raw ingested chunk text directly, bypassing
retrieval and the LLM entirely. Use this to check whether a fact actually
made it into the database in readable form -- if a keyword search here
finds nothing, the problem is PDF text extraction, not retrieval or
generation.

Usage:
    python debug_search.py "26 January"
    python debug_search.py "commencement"
    python debug_search.py "" --page 1        # dump all text from page 1
"""
import argparse
import sys

from db import get_connection


def search_chunks(keyword: str, page: int = None, source: str = None, limit: int = 10):
    conn = get_connection()
    sql = "SELECT chunk_id, document_id, page, text FROM chunks WHERE 1=1"
    params = []

    if keyword:
        sql += " AND text LIKE ?"
        params.append(f"%{keyword}%")
    if page is not None:
        sql += " AND page = ?"
        params.append(page)
    if source:
        sql += """ AND document_id IN (
            SELECT document_id FROM documents WHERE source LIKE ?
        )"""
        params.append(f"%{source}%")

    sql += " LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search raw ingested chunk text directly.")
    parser.add_argument("keyword", nargs="?", default="", help="Text to search for (case-sensitive substring match). Omit to skip.")
    parser.add_argument("--page", type=int, default=None, help="Filter to a specific page number")
    parser.add_argument("--source", default=None, help="Filter to documents whose filename contains this")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    results = search_chunks(args.keyword, args.page, args.source, args.limit)

    if not results:
        print(f"No chunks found matching keyword={args.keyword!r} page={args.page} source={args.source!r}")
        sys.exit(0)

    for row in results:
        print(f"\n--- chunk_id={row['chunk_id']}  page={row['page']} ---")
        print(row["text"][:800])
