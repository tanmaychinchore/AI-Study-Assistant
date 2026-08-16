"""
Astra DB Vector Store API routes.

Provides endpoints for:
  - Checking vector database health and collection configuration
  - Inserting a test chunk (text + BGE-M3 1024-dim embedding + metadata)
  - Retrieving a chunk by ID to verify vector dimension & metadata
  - Deleting a test chunk for cleanup
"""

import time
import uuid

from fastapi import APIRouter, HTTPException, Request, status

from app.core.logging import get_logger
from app.schemas.astra_db import (
    AstraDBChunkResponse,
    AstraDBHealthResponse,
    AstraDBInsertRequest,
    AstraDBInsertResponse,
)
from app.schemas.chunk import DocumentChunk
from app.schemas.document import FileType
from app.schemas.embedding import EmbeddedDocumentChunk
from app.schemas.response import ErrorResponse, SuccessResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/vector-db", tags=["Vector Database"])


def _get_astra_db_service(request: Request):
    """Retrieve AstraDBService from application state."""
    service = getattr(request.app.state, "astra_db_service", None)
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Astra DB service is not connected or collection is not initialized. Please verify ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN in .env.",
        )
    return service


def _get_embedding_service(request: Request):
    """Retrieve EmbeddingService from application state."""
    service = getattr(request.app.state, "embedding_service", None)
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model is not loaded. Service is starting up.",
        )
    return service


@router.get(
    "/health",
    response_model=SuccessResponse,
    summary="Astra DB vector store health check",
    description="Returns connectivity status, keyspace, collection name, and vector configuration for Astra DB.",
)
async def vector_db_health(request: Request) -> SuccessResponse:
    """Return Astra DB vector collection health info."""
    service = getattr(request.app.state, "astra_db_service", None)
    if service is None:
        health_info = {
            "status": "not_configured",
            "keyspace": "",
            "collection": "",
            "vector_dimension": 1024,
            "metric": "cosine",
            "is_connected": False,
            "collection_exists": False,
            "detail": "AstraDBService not instantiated.",
        }
    else:
        health_info = service.get_health()

    return SuccessResponse(
        message="Astra DB health status retrieved.",
        data=health_info,
    )


@router.post(
    "/test-insert",
    response_model=SuccessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Astra DB or Embedding service unavailable"},
    },
    summary="Insert a test chunk into Astra DB",
    description="Embeds the input text using local BGE-M3 (1024-dim) and inserts the vector document into Astra DB.",
)
async def test_insert_chunk(
    request: Request,
    body: AstraDBInsertRequest,
) -> SuccessResponse:
    """Embed and insert a single test chunk into Astra DB."""
    if not body.text or not body.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content cannot be empty.",
        )

    astra_service = _get_astra_db_service(request)
    embedding_service = _get_embedding_service(request)

    try:
        # Generate 1024-dim vector using BGE-M3
        vector = embedding_service.embed_query(body.text)

        # Build DocumentChunk + EmbeddedDocumentChunk
        chunk_id = f"{body.document_id or 'test'}_chunk_{uuid.uuid4().hex[:8]}"
        doc_chunk = DocumentChunk(
            chunk_id=chunk_id,
            chunk_index=0,
            text=body.text,
            char_count=len(body.text),
            document_id=body.document_id or "test_doc_001",
            document_name=body.document_name,
            file_type=FileType.TXT,
            user_id=body.user_id or "test_user",
            subject=body.subject,
            topic=body.topic,
            page_number=body.page_number,
            slide_number=body.slide_number,
            slide_title=body.slide_title,
            heading=body.heading,
            source_type="test",
        )

        embedded_chunk = EmbeddedDocumentChunk.from_chunk_and_vector(doc_chunk, vector)

        # Insert into Astra DB
        inserted_count, inserted_ids, duration_ms = astra_service.insert_embedded_chunks([embedded_chunk])

        response_data = AstraDBInsertResponse(
            inserted_count=inserted_count,
            failed_count=0,
            inserted_ids=inserted_ids,
            duration_ms=duration_ms,
            collection=astra_service.collection_name,
        )

        return SuccessResponse(
            message=f"Test chunk '{chunk_id}' successfully embedded and stored in Astra DB.",
            data=response_data.model_dump(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to insert test chunk into Astra DB")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Insertion failed: {exc}",
        )


@router.get(
    "/test-document/{chunk_id}",
    response_model=SuccessResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Chunk not found"},
        503: {"model": ErrorResponse, "description": "Astra DB service unavailable"},
    },
    summary="Retrieve a chunk by ID from Astra DB",
    description="Fetches stored chunk document, verifies metadata and confirms 1024-dim vector presence.",
)
async def test_get_chunk(
    request: Request,
    chunk_id: str,
) -> SuccessResponse:
    """Retrieve chunk document from Astra DB by ID."""
    astra_service = _get_astra_db_service(request)

    doc = astra_service.get_chunk(chunk_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with ID '{chunk_id}' was not found in Astra DB collection '{astra_service.collection_name}'.",
        )

    return SuccessResponse(
        message=f"Chunk '{chunk_id}' retrieved successfully.",
        data=doc,
    )


@router.delete(
    "/test-document/{chunk_id}",
    response_model=SuccessResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Chunk not found"},
        503: {"model": ErrorResponse, "description": "Astra DB service unavailable"},
    },
    summary="Delete a test chunk by ID from Astra DB",
    description="Removes a test chunk document from Astra DB.",
)
async def test_delete_chunk(
    request: Request,
    chunk_id: str,
) -> SuccessResponse:
    """Delete chunk document from Astra DB."""
    astra_service = _get_astra_db_service(request)

    deleted = astra_service.delete_chunk(chunk_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk '{chunk_id}' not found or could not be deleted.",
        )

    return SuccessResponse(
        message=f"Chunk '{chunk_id}' deleted successfully from Astra DB.",
        data={"chunk_id": chunk_id, "deleted": True},
    )
