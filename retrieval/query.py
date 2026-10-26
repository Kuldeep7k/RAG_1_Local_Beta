"""Query interpretation shared by all retrieval backends."""
import re
from dataclasses import dataclass


_ARTICLE_REFERENCE_RE = re.compile(r"\barticle\s+(\d{1,3}[A-Z]{0,2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class RetrievalConstraints:
    """Structured filters derived from a user query.

    These constraints are applied by every retrieval backend, never after
    candidates have already been retrieved.
    """
    article_ref: str | None = None


def parse_retrieval_constraints(query: str) -> RetrievalConstraints:
    """Extract an explicitly named Constitution article reference, if present."""
    match = _ARTICLE_REFERENCE_RE.search(query)
    return RetrievalConstraints(article_ref=match.group(1).upper() if match else None)
