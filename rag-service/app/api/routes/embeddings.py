"""
Embedding test API routes.

Provides a development/testing endpoint to verify the embedding service
without having to go through the full document pipeline.
"""

import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request, status

from app.core.logging import get_logger
from app.schemas.embedding import EmbeddingTestRequest, EmbeddingTestResponse
from app.schemas.response import ErrorResponse, SuccessResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


from app.core.config import settings

def _get_embedding_service(request: Request):
    """Retrieve the EmbeddingService from application state."""
    if settings.ENVIRONMENT == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Development endpoints are disabled in production.",
        )
    service = getattr(request.app.state, "embedding_service", None)
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model is not loaded. Service is starting up.",
        )
    return service


@router.post(
    "/test",
    response_model=SuccessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Test the embedding service",
    description=(
        "Submit one or more text strings to verify the embedding service is working. "
        "Returns embedding metadata (dimension, device, model) and optionally a "
        "cosine similarity matrix.  Does NOT return full 1024-number vectors — "
        "use the include_similarity flag to see how texts relate to each other."
    ),
)
async def test_embeddings(
    request: Request,
    body: EmbeddingTestRequest,
) -> SuccessResponse:
    """Test the embedding service with sample texts."""
    service = _get_embedding_service(request)

    # Validate texts
    for i, text in enumerate(body.texts):
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text at index {i} is empty or whitespace-only.",
            )

    try:
        start = time.perf_counter()
        vectors = service.embed_texts(body.texts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Build similarity matrix if requested
        similarity_matrix = None
        if body.include_similarity and len(vectors) > 1:
            arr = np.array(vectors)
            # Cosine similarity = dot product of normalized vectors
            sim = np.dot(arr, arr.T)
            similarity_matrix = [[round(float(v), 4) for v in row] for row in sim]

        response_data = EmbeddingTestResponse(
            model=service.model_name,
            device=service.device,
            embedding_dimension=len(vectors[0]),
            count=len(vectors),
            texts=body.texts,
            embedding_time_ms=round(elapsed_ms, 2),
            similarity_matrix=similarity_matrix,
        )

        return SuccessResponse(
            message=f"Successfully embedded {len(vectors)} text(s).",
            data=response_data.model_dump(),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Embedding test failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {exc}",
        )
