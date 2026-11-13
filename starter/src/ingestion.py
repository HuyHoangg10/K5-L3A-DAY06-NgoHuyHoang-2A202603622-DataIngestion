"""PDF ingestion with page-level traceability and conservative normalization."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage


_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\r\n]+")
_MULTIPLE_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_HEADING = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\s+[A-Z]"
    r"|(?:abstract|introduction|conclusion|discussion|references|appendix)\b"
    r"|task\s*#?\d+\s*:"
    r")",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while retaining useful boundaries.

    NFC keeps mathematical symbols and PDF ligatures intact. Wrapped PDF lines are
    joined with spaces, while blank lines, headings, and likely sentence/paragraph
    boundaries are retained as double newlines for structure-aware chunking. Visible
    line-end hyphens are deliberately retained: removing them without layout/OCR
    evidence can silently change technical terms and break gold-evidence matching.
    """
    if not text:
        return ""

    value = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    value = _CONTROL_CHARACTERS.sub("", value)
    raw_lines = value.split("\n")
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            paragraph = " ".join(current).strip()
            if paragraph:
                paragraphs.append(paragraph)
            current.clear()

    previous_line = ""
    for raw_line in raw_lines:
        line = _HORIZONTAL_WHITESPACE.sub(" ", raw_line).strip()
        if not line:
            flush()
            previous_line = ""
            continue

        is_heading = len(line) <= 140 and bool(_HEADING.match(line))
        starts_new_thought = bool(
            current
            and previous_line
            and re.search(r"[.!?][\"'”’)]?$", previous_line)
            and re.match(r"(?:[A-Z0-9\[“]|[A-Z][a-z])", line)
        )
        if is_heading or starts_new_thought:
            flush()
        current.append(line)
        if is_heading:
            flush()
        previous_line = line
    flush()

    normalized = "\n\n".join(paragraphs)
    normalized = _MULTIPLE_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    metadata: dict[str, Any] | None = None,
    issues: list[dict[str, Any]] | None = None,
) -> list[DocumentPage]:
    """Extract every physical page and preserve its 1-based page number.

    Empty pages and page-level extraction errors are recorded in ``issues``. A
    failed page is represented with empty text so the caller can decide whether to
    exclude it, apply OCR, or fail validation; this function never invents text.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    issue_sink = issues if issues is not None else []
    base_metadata = dict(metadata or {})
    reader = PdfReader(pdf_path)
    pages: list[DocumentPage] = []

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        status = "ok"
        error: str | None = None
        try:
            raw_text = pdf_page.extract_text() or ""
            text = normalize_text(raw_text)
            if not text:
                status = "empty"
                issue_sink.append(
                    {
                        "document_id": document_id,
                        "source": source,
                        "page": page_number,
                        "status": status,
                        "message": "No extractable text; OCR or manual review may be required.",
                    }
                )
        except Exception as exc:  # pypdf can fail on a single malformed page
            text = ""
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            issue_sink.append(
                {
                    "document_id": document_id,
                    "source": source,
                    "page": page_number,
                    "status": status,
                    "message": error,
                }
            )

        page_metadata = {
            **base_metadata,
            "extraction_status": status,
            "normalized_characters": len(text),
        }
        if error:
            page_metadata["extraction_error"] = error
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata=page_metadata,
            )
        )

    return pages
