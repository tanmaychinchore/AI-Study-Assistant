"""
Document service — high-level pipeline orchestrator.

Connects the stages:
  extract → clean → chunk

into a single call that the API route can invoke.
Future tasks will extend this to include embedding and indexing.
"""

import time
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger
from app.schemas.chunk import ChunkedDocument
from app.schemas.document import ProcessingStatus
from app.services.extraction_service import extract_document
from app.services.cleaning_service import clean_document
from app.services.chunking_service import chunk_document

logger = get_logger(__name__)


def process_document_pipeline(
    file_path: Path,
    user_id: str,
    document_id: Optional[str] = None,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> ChunkedDocument:
    """
    Run the full document processing pipeline:
    extract → clean → chunk.

    Parameters
    ----------
    file_path : Path
        Path to the uploaded file.
    user_id : str
        Owning user ID.
    document_id : str, optional
        Pre-generated document ID.
    subject : str, optional
        Subject label.
    topic : str, optional
        Topic label.
    chunk_size : int, optional
        Override chunk size (defaults to config).
    chunk_overlap : int, optional
        Override chunk overlap (defaults to config).

    Returns
    -------
    ChunkedDocument
        Fully processed document with chunks ready for embedding.
    """
    pipeline_start = time.perf_counter()

    logger.info(
        "Pipeline started: file=%s  user=%s",
        file_path.name,
        user_id,
    )

    # --- Stage 1: Extract ---
    extracted = extract_document(
        file_path=file_path,
        user_id=user_id,
        document_id=document_id,
        subject=subject,
        topic=topic,
    )
    extraction_time = extracted.extraction_time_ms or 0

    # --- Stage 2: Clean ---
    cleaned, cleaning_time = clean_document(extracted)

    # --- Stage 3: Chunk ---
    chunked, chunking_time = chunk_document(
        cleaned,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # --- Finalize ---
    total_time = (time.perf_counter() - pipeline_start) * 1000

    # Update timing fields
    chunked = chunked.model_copy(
        update={
            "extraction_time_ms": round(extraction_time, 2),
            "cleaning_time_ms": cleaning_time,
            "chunking_time_ms": chunking_time,
            "total_processing_time_ms": round(total_time, 2),
            "status": ProcessingStatus.CHUNKING,
        }
    )

    logger.info(
        "Pipeline complete: doc=%s  pages=%d  chunks=%d  "
        "extract=%.1fms  clean=%.1fms  chunk=%.1fms  total=%.1fms",
        chunked.document_name,
        chunked.total_chunks,
        chunked.total_chunks,
        extraction_time,
        cleaning_time,
        chunking_time,
        total_time,
    )

    return chunked
