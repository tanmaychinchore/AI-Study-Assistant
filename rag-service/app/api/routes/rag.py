"""
RAG query API route.

Provides the end-to-end question answering endpoint that orchestrates
semantic retrieval, context budgeting, and Groq LLM grounded text generation.
"""

from fastapi import APIRouter, HTTPException, Request, status

from app.core.logging import get_logger
from app.schemas.rag import RAGRequest, RAGResult
from app.schemas.response import SuccessResponse
from app.services.groq_service import (
    GroqAuthError,
    GroqModelError,
    GroqRateLimitError,
    GroqServiceError,
    GroqTimeoutError,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/query",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question against indexed study material",
    description=(
        "Executes the end-to-end RAG pipeline: retrieves relevant document chunks via BGE-M3 "
        "and Astra DB vector search, formats budgeted context with source metadata, and generates "
        "a grounded, cited answer using Groq LLM."
    ),
)
async def rag_query_endpoint(
    request_body: RAGRequest,
    request: Request,
) -> SuccessResponse:
    """
    Execute grounded RAG question answering.
    """
    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG generation service is not available on this server.",
        )

    try:
        result: RAGResult = rag_service.query(request_body)

        return SuccessResponse(
            message="RAG query completed successfully.",
            data=result,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except GroqAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except GroqRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc

    except GroqTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc

    except GroqModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except GroqServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error("Unhandled error during RAG query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during RAG generation: {exc}",
        ) from exc
