"""
Retrieval Service.

Orchestrates the complete semantic retrieval pipeline:
  1. Query Validation & Parameter Normalization
  2. BGE-M3 Query Embedding (1024-dimensional normalized vector)
  3. Astra DB Vector Search with Metadata Filtering & User Isolation
  4. Similarity Threshold Post-filtering & Top-K Ranking
  5. Statistics & Result Compilation
"""

import time
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatistics,
    RetrievedChunk,
)
from app.services.astra_db_service import AstraDBService
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class RetrievalService:
    """
    Orchestration service for semantic search and chunk retrieval.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        astra_service: Optional[AstraDBService] = None,
    ):
        self.embedding_service = embedding_service
        self.astra_service = astra_service

    def retrieve(
        self,
        request: RetrievalRequest,
        embedding_service: Optional[EmbeddingService] = None,
        astra_service: Optional[AstraDBService] = None,
    ) -> RetrievalResult:
        """
        Execute the end-to-end retrieval pipeline for a user query.

        Parameters
        ----------
        request : RetrievalRequest
            Search query, user ID, top_k, and optional metadata filters.
        embedding_service : EmbeddingService, optional
            Override or fallback embedding service.
        astra_service : AstraDBService, optional
            Override or fallback Astra DB service.

        Returns
        -------
        RetrievalResult
            Ranked chunks, similarity scores, metadata, and timing statistics.
        """
        overall_start = time.perf_counter()

        emb_svc = embedding_service or self.embedding_service
        if emb_svc is None or not emb_svc.is_loaded:
            raise RuntimeError("EmbeddingService is not available or model is not loaded.")

        db_svc = astra_service or self.astra_service
        if db_svc is None or not db_svc.is_ready:
            raise RuntimeError("AstraDBService is not connected or vector collection is not initialized.")

        # --- Validate top_k bounds ---
        top_k = request.top_k
        if top_k < settings.MIN_TOP_K:
            top_k = settings.MIN_TOP_K
        elif top_k > settings.MAX_TOP_K:
            top_k = settings.MAX_TOP_K

        logger.info(
            "=== Starting Retrieval: query='%s'  user_id='%s'  top_k=%d ===",
            request.query[:80] + ("..." if len(request.query) > 80 else ""),
            request.user_id,
            top_k,
        )

        # ------------------------------------------------------------------
        # Stage 1: Query Embedding
        # ------------------------------------------------------------------
        emb_start = time.perf_counter()
        query_vector = emb_svc.embed_query(request.query)
        embedding_time_ms = round((time.perf_counter() - emb_start) * 1000, 2)

        logger.info(
            "[Retrieval Stage 1/2] Query embedded: dim=%d  time=%.1fms",
            len(query_vector),
            embedding_time_ms,
        )

        # ------------------------------------------------------------------
        # Stage 2: Astra DB Vector Search
        # ------------------------------------------------------------------
        raw_chunks, search_time_ms, chunks_retrieved = db_svc.vector_search(
            query_vector=query_vector,
            user_id=request.user_id,
            top_k=top_k,
            document_id=request.document_id,
            subject=request.subject,
            topic=request.topic,
            similarity_threshold=request.similarity_threshold,
        )

        # Build list of RetrievedChunk models
        retrieved_chunks = [RetrievedChunk(**chunk_data) for chunk_data in raw_chunks]

        total_time_ms = round((time.perf_counter() - overall_start) * 1000, 2)

        # Collect applied filters for transparency
        filters_applied: dict[str, Any] = {"user_id": request.user_id}
        if request.document_id:
            filters_applied["document_id"] = request.document_id
        if request.subject:
            filters_applied["subject"] = request.subject
        if request.topic:
            filters_applied["topic"] = request.topic
        if request.similarity_threshold is not None:
            filters_applied["similarity_threshold"] = request.similarity_threshold

        stats = RetrievalStatistics(
            embedding_time_ms=embedding_time_ms,
            search_time_ms=search_time_ms,
            total_time_ms=total_time_ms,
            chunks_retrieved=chunks_retrieved,
            chunks_returned=len(retrieved_chunks),
        )

        logger.info(
            "=== Retrieval Complete: returned=%d/%d  time=%.1fms ===",
            len(retrieved_chunks),
            chunks_retrieved,
            total_time_ms,
        )

        return RetrievalResult(
            query=request.query,
            user_id=request.user_id,
            top_k=top_k,
            filters_applied=filters_applied,
            results=retrieved_chunks,
            statistics=stats,
        )


def retrieve_chunks(
    request: RetrievalRequest,
    embedding_service: Optional[EmbeddingService] = None,
    astra_service: Optional[AstraDBService] = None,
) -> RetrievalResult:
    """
    Convenience function to execute retrieval without manually managing a service instance.
    """
    service = RetrievalService(
        embedding_service=embedding_service,
        astra_service=astra_service,
    )
    return service.retrieve(request)
