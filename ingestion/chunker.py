"""Recursive chunking: splits page text into overlapping, token-bounded
chunks while trying to preserve paragraph (and sentence, for oversized
paragraphs) boundaries rather than cutting mid-thought."""
import re
import bisect
from typing import List, Dict, Optional

# Matches a Constitution article heading at a line boundary. Article
# references are identifiers, not integers: the Constitution includes 21A,
# 239AA, and similar inserted articles. Keeping the complete identifier is
# essential because it is used as a retrieval filter.
_ARTICLE_HEADING_RE = re.compile(
    r'(?m)^\s*(?:[\uf02a*]*\d+\[|\[)?(?P<ref>\d{1,3}[A-Z]{0,2})\.\s+'
    # Amendment footnotes are also numbered lines (for example, "2. Subs.
    # by ..."). They are not article headings and must never become a
    # boundary that absorbs the following article.
    r'(?!(?:Ins\.|Subs\.|Omitted\.?|Added\.?|Rep\.|The\s+(?:words|figures)|For\s+the|Cl\.|Entry\.))'
    # pypdf sometimes emits a replacement character (\ufffd) in place of
    # the em dash used by this official PDF. Treat it as a heading delimiter,
    # but only after a numbered, title-cased article heading.
    # A wrapped title may continue on the next line, but it may never cross
    # another numbered heading. This prevents a numbered amendment footnote
    # from absorbing the next real Article's delimiter.
    r'(?P<title>(?:\d+\[)*\[?[A-Z](?:(?!\n\s*(?:[\uf02a*]*\d+\[|\[)?\d{1,3}[A-Z]{0,2}\.\s+)[\s\S]){3,180}?)(?:\]\s*)?\.?\s*[\u2014\u2013\ufffd]'
)
_OMITTED_ARTICLE_HEADING_RE = re.compile(
    r'(?mi)^\s*(?:[\uf02a*]*\d+\[|\[)?(?P<ref>\d{1,3}[A-Z]{0,2})\.\s+'
    r'(?P<title>[^\n]{0,140}?\bOmitted\.?)\]?\s*$'
)
_CONSTITUTION_PREAMBLE_RE = re.compile(r'\bWE,\s*THE\s*PEOPLE\s+OF\s+INDIA\b', re.IGNORECASE)
# The real Schedule heading is followed by its bracketed constitutional
# references (for example, "[Articles 1 and 4]"). Requiring that structure
# avoids matching ordinary Article text such as "Part B of the First
# Schedule".
_FIRST_SCHEDULE_RE = re.compile(
    r'(?mi)^\s*(?:\d+\[)?FIRST\s+SCHEDULE\s*\r?\n\s*\['
)


def detect_article_ref(text: str) -> Optional[str]:
    """Return a normalized article identifier from a chunk heading, if any."""
    for pattern in (_ARTICLE_HEADING_RE, _OMITTED_ARTICLE_HEADING_RE):
        match = pattern.search(text[:300])
        if match:
            return match.group("ref")
    return None


def detect_article_number(text: str) -> Optional[int]:
    """Compatibility helper for consumers that only support numeric articles."""
    article_ref = detect_article_ref(text)
    return int(article_ref) if article_ref and article_ref.isdigit() else None


def _split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n") if p.strip()]


def _split_into_sentences(text: str) -> List[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?।])\s+", text)
        if sentence.strip()
    ]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_oversized_text(text: str, chunk_size_tokens: int) -> List[str]:
    """Split a sentence that still exceeds the limit without cutting words."""
    words = text.split()
    pieces = []
    current = []
    current_tokens = 0
    for word in words:
        word_tokens = _estimate_tokens(word + " ")
        if current and current_tokens + word_tokens > chunk_size_tokens:
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _tail_by_tokens(text: str, token_count: int) -> str:
    """Return a word-aligned tail for overlap instead of slicing characters."""
    if token_count <= 0:
        return ""
    words = text.split()
    selected = []
    total = 0
    for word in reversed(words):
        total += _estimate_tokens(word + " ")
        selected.append(word)
        if total >= token_count:
            break
    return " ".join(reversed(selected))


def chunk_page(
    text: str,
    chunk_size_tokens: int = 220,
    overlap_tokens: int = 35,
) -> List[str]:
    if chunk_size_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk size must be positive and overlap must be smaller than chunk size")
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        completed = " ".join(current)
        chunks.append(completed)
        overlap_text = _tail_by_tokens(completed, overlap_tokens)
        current = [overlap_text] if overlap_text else []
        current_tokens = _estimate_tokens(overlap_text) if overlap_text else 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        if para_tokens > chunk_size_tokens:
            for sentence in _split_into_sentences(para):
                pieces = _split_oversized_text(sentence, chunk_size_tokens)
                for piece in pieces:
                    piece_tokens = _estimate_tokens(piece)
                    if current_tokens + piece_tokens > chunk_size_tokens and current:
                        flush()
                    current.append(piece)
                    current_tokens += piece_tokens
            continue

        if current_tokens + para_tokens > chunk_size_tokens and current:
            flush()

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _build_offset_page_map(pages: List[Dict]):
    """Concatenates all page texts into one string for whole-document
    boundary detection, tracking which page each character offset came
    from (an article can start on one page and continue onto the next)."""
    parts = []
    offsets = []
    page_numbers = []
    cursor = 0
    for page_data in pages:
        offsets.append(cursor)
        page_numbers.append(page_data["page"])
        text = page_data["text"]
        parts.append(text)
        cursor += len(text) + 1  # +1 for the "\n" join below
    return "\n".join(parts), offsets, page_numbers


def _page_for_offset(offset: int, offsets: List[int], page_numbers: List[int]) -> int:
    idx = bisect.bisect_right(offsets, offset) - 1
    idx = max(0, min(idx, len(page_numbers) - 1))
    return page_numbers[idx]


def chunk_document(
    pages: List[Dict],
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 75,
) -> List[Dict]:
    """Structure-aware chunking: finds every article-heading boundary across
    the WHOLE document first (not per-page, since an article can span a page
    break), splits into one segment per article, then token-chunks only
    WITHIN each segment. This guarantees a chunk never mixes content from
    two different articles.

    Falls back to plain per-page token chunking when no article headings
    are detected at all (e.g. non-legal documents), so this is safe as the
    default chunker for any document type.

    Returns [{text, page, chunk_index, article_ref}, ...]. chunk_index
    is a document-wide running counter so chunk IDs stay unique even when
    two articles start on the same page.
    """
    if chunk_size_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk size must be positive and overlap must be smaller than chunk size")
    if not pages:
        return []
    full_text, offsets, page_numbers = _build_offset_page_map(pages)
    # The contents pages contain article numbers but are not legal text. For
    # the Constitution, the preamble reliably marks the beginning of the body.
    # Other document types fall back to the beginning of their extracted text.
    preamble = _CONSTITUTION_PREAMBLE_RE.search(full_text)
    body_start = preamble.start() if preamble else 0
    first_schedule = _FIRST_SCHEDULE_RE.search(full_text, pos=body_start)
    article_body_end = first_schedule.start() if first_schedule else len(full_text)
    matches_by_offset = {}
    for pattern in (_ARTICLE_HEADING_RE, _OMITTED_ARTICLE_HEADING_RE):
        for match in pattern.finditer(full_text, pos=body_start, endpos=article_body_end):
            matches_by_offset.setdefault(match.start(), match)
    matches = [matches_by_offset[offset] for offset in sorted(matches_by_offset)]

    all_chunks: List[Dict] = []
    counter = [0]

    def emit_segment(segment_text: str, start_offset: int, article_ref: Optional[str]):
        page = _page_for_offset(start_offset, offsets, page_numbers)
        for sub_text in chunk_page(segment_text, chunk_size_tokens, overlap_tokens):
            tagged = f"[Article {article_ref}] {sub_text}" if article_ref is not None else sub_text
            all_chunks.append({
                "text": tagged,
                "page": page,
                "chunk_index": counter[0],
                "article_ref": article_ref,
                "article_number": int(article_ref) if article_ref and article_ref.isdigit() else None,
            })
            counter[0] += 1

    if not matches:
        for i, page_data in enumerate(pages):
            emit_segment(page_data["text"], offsets[i], None)
        return all_chunks

    if matches[0].start() > 0:
        front_text = full_text[: matches[0].start()].strip()
        if front_text:
            emit_segment(front_text, 0, None)

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else article_body_end
        segment_text = full_text[start:end].strip()
        emit_segment(segment_text, start, match.group("ref"))

    # Schedule paragraphs have their own numbering and must not be mistaken
    # for constitutional Articles. They remain searchable as ordinary chunks.
    if first_schedule:
        for index, page_data in enumerate(pages):
            page_offset = offsets[index]
            page_end = page_offset + len(page_data["text"])
            if page_end <= article_body_end:
                continue
            schedule_text = page_data["text"]
            if page_offset < article_body_end:
                schedule_text = schedule_text[article_body_end - page_offset:]
            if schedule_text.strip():
                emit_segment(schedule_text, max(page_offset, article_body_end), None)

    return all_chunks
