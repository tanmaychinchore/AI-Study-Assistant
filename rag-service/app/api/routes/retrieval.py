"""
Retrieval API routes.

Provides semantic search endpoints for querying indexed study documents.
"""

from fastapi import APIRouter, HTTPException, Request, status

from app.core.logging import get_logger
from app.schemas.response import SuccessResponse
from app.schemas.retrieval import RetrievalRequest, RetrievalResult
from app.services.retrieval_service import retrieve_chunks

logger = get_logger(__name__)

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.post(
    "/search",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Search relevant document chunks by natural-language query",
    description=(
        "Embeds the input query with BGE-M3, executes cosine vector similarity search "
        "in Astra DB, applies user isolation and optional metadata filters, and returns "
        "ranked document chunks with similarity scores."
    ),
)
async def search_retrieval_endpoint(
    request_body: RetrievalRequest,
    request: Request,
) -> SuccessResponse:
    """
    Execute semantic similarity search for study document chunks.
    """
    # ------------------------------------------------------------------
    # 1. Verify service readiness from application state
    # ------------------------------------------------------------------
    embedding_service = getattr(request.app.state, "embedding_service", None)
    if embedding_service is None or not embedding_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is not initialized or model is not loaded.",
        )

    astra_service = getattr(request.app.state, "astra_db_service", None)
    if astra_service is None or not astra_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Astra DB service is not connected or vector collection is not ready.",
        )

    # ------------------------------------------------------------------
    # 2. Execute retrieval pipeline
    # ------------------------------------------------------------------
    try:
        result = retrieve_chunks(
            request=request_body,
            embedding_service=embedding_service,
            astra_service=astra_service,
        )

        match_count = len(result.results)
        msg = (
            f"Retrieved {match_count} relevant chunk(s) for query."
            if match_count > 0
            else "No chunks matched the search criteria or similarity threshold."
        )

        return SuccessResponse(
            success=True,
            message=msg,
            data=result,
        )

    except ValueError as exc:
        logger.warning("Retrieval validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.error("Retrieval runtime error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error("Unexpected error during retrieval: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during retrieval: {exc}",
        ) from exc
