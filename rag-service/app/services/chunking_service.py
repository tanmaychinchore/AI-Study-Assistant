"""
Document chunking service.

Takes a cleaned ExtractedDocument and splits it into small, overlapping
chunks suitable for embedding and vector storage.

Uses LangChain's RecursiveCharacterTextSplitter which tries to keep
paragraphs, sentences, and words together before falling back to
character-level splitting.
"""

import time
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.chunk import ChunkedDocument, DocumentChunk
from app.schemas.document import ExtractedDocument, ExtractedPage, ProcessingStatus

logger = get_logger(__name__)


def _build_chunk_id(document_id: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    return f"{document_id}_chunk_{index:04d}"


def chunk_document(
    document: ExtractedDocument,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> tuple[ChunkedDocument, float]:
    """
    Split a cleaned document into chunks with metadata.

    Each chunk inherits the source metadata (page/slide/heading) from the
    page it came from, so the RAG pipeline can later cite exact sources.

    Parameters
    ----------
    document : ExtractedDocument
        Cleaned document to chunk.
    chunk_size : int, optional
        Maximum characters per chunk (defaults to settings.CHUNK_SIZE).
    chunk_overlap : int, optional
        Overlap characters between chunks (defaults to settings.CHUNK_OVERLAP).

    Returns
    -------
    tuple[ChunkedDocument, float]
        The chunked document and the chunking time in milliseconds.
    """
    start = time.perf_counter()

    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    # Ensure overlap is always less than size (LangChain requirement)
    if overlap >= size:
        overlap = size // 5

    logger.info(
        "Chunking started: doc=%s  pages=%d  chunk_size=%d  overlap=%d",
        document.document_name,
        len(document.pages),
        size,
        overlap,
    )

    # --- Configure the splitter ---
    # RecursiveCharacterTextSplitter tries to split on these separators
    # in order, keeping larger semantic units together:
    #   "\n\n" (paragraphs) → "\n" (lines) → " " (words) → "" (characters)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        is_separator_regex=False,
    )

    # --- Split each page and propagate metadata ---
    all_chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in document.pages:
        if not page.text or not page.text.strip():
            continue

        # Split the page text into pieces
        text_pieces = splitter.split_text(page.text)

        for piece in text_pieces:
            piece = piece.strip()
            if not piece:
                continue

            chunk = DocumentChunk(
                chunk_id=_build_chunk_id(document.document_id, chunk_index),
                chunk_index=chunk_index,
                text=piece,
                char_count=len(piece),
                # Source identity
                document_id=document.document_id,
                document_name=document.document_name,
                file_type=document.file_type,
                user_id=document.user_id,
                subject=document.subject,
                topic=document.topic,
                # Page/slide source
                page_number=page.page_number,
                slide_number=page.slide_number,
                slide_title=page.slide_title,
                heading=page.heading,
            )

            all_chunks.append(chunk)
            chunk_index += 1

    total_chars = sum(c.char_count for c in all_chunks)
    elapsed_ms = (time.perf_counter() - start) * 1000

    chunked_doc = ChunkedDocument(
        document_id=document.document_id,
        document_name=document.document_name,
        file_type=document.file_type,
        user_id=document.user_id,
        subject=document.subject,
        topic=document.topic,
        chunks=all_chunks,
        total_chunks=len(all_chunks),
        total_characters=total_chars,
        chunk_size=size,
        chunk_overlap=overlap,
        status=ProcessingStatus.CHUNKING,
    )

    logger.info(
        "Chunking complete: doc=%s  chunks=%d  total_chars=%d  time=%.2fms",
        document.document_name,
        len(all_chunks),
        total_chars,
        elapsed_ms,
    )

    return chunked_doc, round(elapsed_ms, 2)
