"""Deterministic fixed-size and structure-aware chunking."""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, DocumentPage


def _chunk_id(strategy: str, page: DocumentPage, index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{strategy}:{page.document_id}:p{page.page:04d}:c{index:04d}:{digest}"


def _metadata(
    page: DocumentPage,
    strategy: str,
    index: int,
    char_start: int,
    char_end: int,
) -> dict[str, object]:
    return {
        **page.metadata,
        "strategy": strategy,
        "chunk_index": index,
        "char_start": char_start,
        "char_end": char_end,
        "chunk_characters": char_end - char_start,
    }


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into deterministic character windows with overlap.

    Window ends and subsequent starts move to nearby whitespace when possible so
    words are not cut. Chunks never cross a physical page boundary.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    text = page.text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        hard_end = min(start + size, len(text))
        end = hard_end
        if hard_end < len(text):
            whitespace = text.rfind(" ", start + max(1, size // 2), hard_end + 1)
            if whitespace > start:
                end = whitespace

        chunk_text = text[start:end].strip()
        leading_trim = len(text[start:end]) - len(text[start:end].lstrip())
        trailing_trim = len(text[start:end]) - len(text[start:end].rstrip())
        actual_start = start + leading_trim
        actual_end = end - trailing_trim
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id("A", page, index, chunk_text),
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    metadata=_metadata(page, "A", index, actual_start, actual_end),
                )
            )
            index += 1

        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        if next_start > 0 and not text[next_start - 1].isspace():
            next_space = text.find(" ", next_start, end)
            if next_space != -1:
                next_start = next_space + 1
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _split_oversized_unit(text: str, max_size: int) -> list[str]:
    """Split a long paragraph at sentence/word boundaries without overlap."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[“])", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_size:
            if current:
                pieces.append(current)
                current = ""
            cursor = 0
            while cursor < len(sentence):
                end = min(cursor + max_size, len(sentence))
                if end < len(sentence):
                    whitespace = sentence.rfind(" ", cursor + max_size // 2, end + 1)
                    if whitespace > cursor:
                        end = whitespace
                piece = sentence[cursor:end].strip()
                if piece:
                    pieces.append(piece)
                cursor = end
                while cursor < len(sentence) and sentence[cursor].isspace():
                    cursor += 1
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_size:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group complete normalized paragraphs up to ``max_size`` characters.

    Oversized paragraphs are split at sentence boundaries, then at word boundaries
    only when a single sentence is itself too long. Chunks remain page-scoped.
    """
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    text = page.text.strip()
    if not text:
        return []

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        units.extend(_split_oversized_unit(paragraph, max_size))

    grouped: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if current and len(candidate) > max_size:
            grouped.append(current)
            current = unit
        else:
            current = candidate
    if current:
        grouped.append(current)

    chunks: list[Chunk] = []
    search_from = 0
    for index, chunk_text in enumerate(grouped):
        # Paragraph whitespace can be normalized while grouping, so locate the first
        # distinctive prefix when an exact substring is unavailable.
        start = text.find(chunk_text, search_from)
        if start < 0:
            prefix = chunk_text[: min(80, len(chunk_text))]
            start = text.find(prefix, search_from)
        if start < 0:
            start = search_from
        end = min(start + len(chunk_text), len(text))
        search_from = end
        chunks.append(
            Chunk(
                chunk_id=_chunk_id("B", page, index, chunk_text),
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=chunk_text,
                metadata=_metadata(page, "B", index, start, end),
            )
        )
    return chunks
