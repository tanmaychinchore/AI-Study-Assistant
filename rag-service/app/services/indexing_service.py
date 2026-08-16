"""
Document Indexing Service — End-to-End Orchestrator (Task 6).

Coordinates the complete 5-stage document indexing pipeline:
  1. Extraction (PyMuPDF, python-pptx, python-docx, native txt)
  2. Cleaning & Normalization (whitespace/structure preservation)
  3. Semantic Chunking (RecursiveCharacterTextSplitter: 1000/200)
  4. Vector Embedding (BAAI/bge-m3: 1024-dim normalized vectors)
  5. Astra DB Vector Storage (Duplicate cleanup + batch insert)

Reuses all existing modular services without duplicating business logic.
Guarantees metadata preservation, duplicate prevention on re-index,
and precise stage-by-stage execution metrics.
"""

from pathlib import Path
import time
from typing import Optional

from app.core.logging import get_logger
from app.schemas.chunk import DocumentChunkPreview
from app.schemas.document import ProcessingStatus
from app.schemas.indexing import IndexingResult, IndexingStatistics
from app.services.astra_db_service import AstraDBService
from app.services.chunking_service import chunk_document
from app.services.cleaning_service import clean_document
from app.services.embedding_service import EmbeddingService
from app.services.extraction_service import extract_document

logger = get_logger(__name__)

CHUNK_PREVIEW_COUNT = 5


class IndexingService:
    """
    Orchestrates the end-to-end document indexing lifecycle.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        astra_service: Optional[AstraDBService] = None,
    ):
        self.embedding_service = embedding_service
        self.astra_service = astra_service

    def index_document(
        self,
        file_path: Path,
        user_id: str,
        document_id: Optional[str] = None,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_service: Optional[EmbeddingService] = None,
        astra_service: Optional[AstraDBService] = None,
    ) -> IndexingResult:
        """
        Execute the full 5-stage indexing pipeline for a document file.

        Parameters
        ----------
        file_path : Path
            Path to the document on disk.
        user_id : str
            ID of the user who owns the document.
        document_id : str, optional
            Pre-assigned document ID. If None, auto-generated.
        subject : str, optional
            Subject tag.
        topic : str, optional
            Topic tag.
        chunk_size : int, optional
            Chunk size in characters (defaults to settings.CHUNK_SIZE).
        chunk_overlap : int, optional
            Chunk overlap in characters (defaults to settings.CHUNK_OVERLAP).
        embedding_service : EmbeddingService, optional
            Override or fallback embedding service.
        astra_service : AstraDBService, optional
            Override or fallback Astra DB service.

        Returns
        -------
        IndexingResult
            Full summary of the indexed document, counts, and timing statistics.
        """
        emb_svc = embedding_service or self.embedding_service
        if emb_svc is None or not emb_svc.is_loaded:
            raise RuntimeError("EmbeddingService is not available or model is not loaded.")

        db_svc = astra_service or self.astra_service
        if db_svc is None or not db_svc.is_ready:
            raise RuntimeError("AstraDBService is not connected or vector collection is not initialized.")

        total_start = time.perf_counter()

        logger.info(
            "=== Starting Document Indexing: file='%s'  user='%s' ===",
            file_path.name,
            user_id,
        )

        # ------------------------------------------------------------------
        # Stage 1: Document Extraction
        # ------------------------------------------------------------------
        logger.info("[Stage 1/5] Extracting text and structure from '%s'...", file_path.name)
        extracted = extract_document(
            file_path=file_path,
            user_id=user_id,
            document_id=document_id,
            subject=subject,
            topic=topic,
        )
        extraction_time_ms = round(extracted.extraction_time_ms or 0.0, 2)
        doc_id = extracted.document_id

        logger.info(
            "[Stage 1/5] Extraction complete: doc_id='%s'  pages=%d  chars=%d  time=%.1fms",
            doc_id,
            extracted.total_pages,
            extracted.total_characters,
            extraction_time_ms,
        )

        # ------------------------------------------------------------------
        # Stage 2: Document Cleaning
        # ------------------------------------------------------------------
        logger.info("[Stage 2/5] Cleaning and normalizing extracted text for '%s'...", doc_id)
        cleaned, cleaning_time_ms = clean_document(extracted)
        cleaning_time_ms = round(cleaning_time_ms, 2)

        logger.info(
            "[Stage 2/5] Cleaning complete: doc_id='%s'  valid_pages=%d  chars=%d  time=%.1fms",
            doc_id,
            cleaned.total_pages,
            cleaned.total_characters,
            cleaning_time_ms,
        )

        # ------------------------------------------------------------------
        # Stage 3: Semantic Chunking
        # ------------------------------------------------------------------
        logger.info("[Stage 3/5] Splitting document '%s' into chunks...", doc_id)
        chunked, chunking_time_ms = chunk_document(
            cleaned,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunking_time_ms = round(chunking_time_ms, 2)

        if not chunked.chunks:
            raise ValueError(f"Document '{file_path.name}' produced 0 chunks after cleaning.")

        logger.info(
            "[Stage 3/5] Chunking complete: doc_id='%s'  chunks=%d  chars=%d  time=%.1fms",
            doc_id,
            chunked.total_chunks,
            chunked.total_characters,
            chunking_time_ms,
        )

        # ------------------------------------------------------------------
        # Stage 4: BGE-M3 Vector Embedding
        # ------------------------------------------------------------------
        logger.info("[Stage 4/5] Generating 1024-dim BGE-M3 embeddings for %d chunk(s)...", len(chunked.chunks))
        embedded_chunks, embedding_time_ms = emb_svc.embed_chunks(chunked.chunks)
        embedding_time_ms = round(embedding_time_ms, 2)

        logger.info(
            "[Stage 4/5] Embedding complete: doc_id='%s'  vectors=%d  dim=%d  time=%.1fms",
            doc_id,
            len(embedded_chunks),
            emb_svc.embedding_dimension,
            embedding_time_ms,
        )

        # ------------------------------------------------------------------
        # Stage 5: Astra DB Vector Storage (with Re-index Duplicate Cleanup)
        # ------------------------------------------------------------------
        logger.info("[Stage 5/5] Persisting vector documents to Astra DB collection '%s'...", db_svc.collection_name)

        # Re-indexing cleanup: remove any existing vectors for this document_id
        db_svc.delete_document_chunks(doc_id)

        # Batch insert new chunks
        inserted_count, inserted_ids, astra_insertion_time_ms = db_svc.insert_embedded_chunks(embedded_chunks)
        astra_insertion_time_ms = round(astra_insertion_time_ms, 2)

        if inserted_count != len(embedded_chunks):
            raise RuntimeError(
                f"Astra DB inserted count mismatch: expected {len(embedded_chunks)}, inserted {inserted_count}"
            )

        logger.info(
            "[Stage 5/5] Astra DB storage complete: doc_id='%s'  inserted=%d  time=%.1fms",
            doc_id,
            inserted_count,
            astra_insertion_time_ms,
        )

        # ------------------------------------------------------------------
        # Finalize & Summarize
        # ------------------------------------------------------------------
        total_time_ms = round((time.perf_counter() - total_start) * 1000, 2)

        # Build chunk previews (first N chunks)
        previews = [
            DocumentChunkPreview(
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                char_count=c.char_count,
                text_preview=c.text[:200] + ("..." if len(c.text) > 200 else ""),
                page_number=c.page_number,
                slide_number=c.slide_number,
                slide_title=c.slide_title,
                heading=c.heading,
            )
            for c in chunked.chunks[:CHUNK_PREVIEW_COUNT]
        ]

        stats = IndexingStatistics(
            extraction_time_ms=extraction_time_ms,
            cleaning_time_ms=cleaning_time_ms,
            chunking_time_ms=chunking_time_ms,
            embedding_time_ms=embedding_time_ms,
            astra_insertion_time_ms=astra_insertion_time_ms,
            total_time_ms=total_time_ms,
        )

        result = IndexingResult(
            document_id=doc_id,
            document_name=chunked.document_name,
            file_type=chunked.file_type,
            user_id=user_id,
            subject=chunked.subject,
            topic=chunked.topic,
            total_pages=extracted.total_pages,
            total_chunks=chunked.total_chunks,
            total_characters=chunked.total_characters,
            embeddings_generated=len(embedded_chunks),
            vectors_inserted=inserted_count,
            collection=db_svc.collection_name,
            status=ProcessingStatus.INDEXED,
            statistics=stats,
            chunks_preview=previews,
        )

        logger.info(
            "=== Document Indexing Successful: doc_id='%s'  chunks=%d  total_time=%.1fms ===",
            doc_id,
            chunked.total_chunks,
            total_time_ms,
        )

        return result


# Singleton instance helper
def index_document(
    file_path: Path,
    user_id: str,
    document_id: Optional[str] = None,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_service: Optional[EmbeddingService] = None,
    astra_service: Optional[AstraDBService] = None,
) -> IndexingResult:
    """Convenience function for indexing a document."""
    service = IndexingService(
        embedding_service=embedding_service,
        astra_service=astra_service,
    )
    return service.index_document(
        file_path=file_path,
        user_id=user_id,
        document_id=document_id,
        subject=subject,
        topic=topic,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
