"""Document loaders: extract text + page-level metadata from source files."""
from pathlib import Path
from typing import List, Dict

import pypdf
import docx


def load_pdf(file_path: str) -> List[Dict]:
    """Returns a list of {text, page} dicts, one per PDF page."""
    reader = pypdf.PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "page": i})
    return pages


def load_docx(file_path: str) -> List[Dict]:
    """Returns a list of {text, page} dicts. DOCX has no real page concept,
    so paragraphs are grouped into pseudo-pages for citation purposes."""
    document = docx.Document(file_path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]

    pages = []
    group_size = 40
    for i in range(0, len(paragraphs), group_size):
        group = paragraphs[i:i + group_size]
        pages.append({"text": "\n".join(group), "page": (i // group_size) + 1})
    return pages


def load_txt(file_path: str) -> List[Dict]:
    try:
        text = Path(file_path).read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Text files must be valid UTF-8") from exc
    return [{"text": text, "page": 1}] if text else []


def load_document(file_path: str) -> List[Dict]:
    """Dispatches to the right loader based on file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        pages = load_pdf(file_path)
    elif suffix == ".docx":
        pages = load_docx(file_path)
    elif suffix in (".txt", ".md"):
        pages = load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if not pages:
        raise ValueError(
            "No extractable text was found. This may be a scanned document; "
            "run OCR before uploading it."
        )
    return pages
