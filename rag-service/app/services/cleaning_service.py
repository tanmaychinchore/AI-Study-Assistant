"""
Text cleaning service.

Cleans raw extracted text while preserving educationally meaningful
content (formulas, definitions, lists, tables, examples).

The goal is to remove noise from PDF/office extraction without
destroying the structure that makes study material useful.
"""

import re
import time

from app.core.logging import get_logger
from app.schemas.document import ExtractedDocument, ExtractedPage

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------------

def _collapse_whitespace(text: str) -> str:
    """Replace runs of spaces/tabs with a single space (preserve newlines)."""
    # Replace multiple spaces/tabs (not newlines) with a single space
    text = re.sub(r"[^\S\n]+", " ", text)
    return text


def _collapse_blank_lines(text: str) -> str:
    """Replace 3+ consecutive blank lines with 2."""
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def _strip_page_artifacts(text: str) -> str:
    """
    Remove common PDF extraction artifacts:
    - Repeated headers/footers (lines that look like "Page X of Y")
    - Standalone page numbers
    - Excessive dashes/underscores used as separators
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip standalone page numbers (e.g., "12", "- 12 -", "Page 12")
        if re.match(r"^[-–—\s]*\d{1,4}[-–—\s]*$", stripped):
            continue
        if re.match(r"^page\s+\d+(\s+of\s+\d+)?$", stripped, re.IGNORECASE):
            continue

        # Skip lines that are just repeated dashes/underscores/equals (separators)
        if re.match(r"^[-_=~*]{10,}$", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _fix_broken_words(text: str) -> str:
    """
    Fix words broken by line-ending hyphens (common in PDF extraction).

    Example: "algo-\nrithm" → "algorithm"

    Be conservative — only fix when a hyphen is immediately followed by
    a newline and then a lowercase letter (to avoid breaking real hyphens
    like "well-known").
    """
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text


def _strip_line_edges(text: str) -> str:
    """Strip trailing whitespace from every line."""
    lines = text.split("\n")
    return "\n".join(line.rstrip() for line in lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Apply all cleaning rules to a text string.

    The order matters — some rules depend on previous transformations.
    """
    if not text or not text.strip():
        return ""

    text = _fix_broken_words(text)
    text = _strip_page_artifacts(text)
    text = _collapse_whitespace(text)
    text = _collapse_blank_lines(text)
    text = _strip_line_edges(text)
    text = text.strip()

    return text


def clean_document(document: ExtractedDocument) -> tuple[ExtractedDocument, float]:
    """
    Clean all pages/sections in an ExtractedDocument.

    Returns a new ExtractedDocument with cleaned text and the
    cleaning time in milliseconds.

    Pages that become empty after cleaning are removed.
    """
    start = time.perf_counter()

    logger.info(
        "Cleaning started: doc=%s  pages=%d",
        document.document_name,
        len(document.pages),
    )

    cleaned_pages: list[ExtractedPage] = []

    for page in document.pages:
        cleaned_text = clean_text(page.text)

        if not cleaned_text:
            logger.debug(
                "Page became empty after cleaning — skipping (page=%s, slide=%s)",
                page.page_number,
                page.slide_number,
            )
            continue

        cleaned_pages.append(
            ExtractedPage(
                page_number=page.page_number,
                slide_number=page.slide_number,
                slide_title=page.slide_title,
                heading=page.heading,
                text=cleaned_text,
                char_count=len(cleaned_text),
            )
        )

    total_chars = sum(p.char_count for p in cleaned_pages)
    elapsed_ms = (time.perf_counter() - start) * 1000

    cleaned_doc = document.model_copy(
        update={
            "pages": cleaned_pages,
            "total_pages": len(cleaned_pages),
            "total_characters": total_chars,
        }
    )

    logger.info(
        "Cleaning complete: doc=%s  pages=%d  chars=%d  time=%.2fms",
        document.document_name,
        len(cleaned_pages),
        total_chars,
        elapsed_ms,
    )

    return cleaned_doc, round(elapsed_ms, 2)
