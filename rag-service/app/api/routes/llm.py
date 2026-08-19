"""
LLM test & development API route.

Provides an endpoint for validating Groq LLM connectivity, model responses,
token usage, and latency. For development and testing purposes only.
"""

from fastapi import APIRouter, HTTPException, Request, status

from app.core.logging import get_logger
from app.schemas.llm import GenerationResult, LLMTestRequest
from app.schemas.response import SuccessResponse
from app.services.groq_service import (
    GroqAuthError,
    GroqModelError,
    GroqRateLimitError,
    GroqServiceError,
    GroqTimeoutError,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])


@router.post(
    "/test",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Test Groq LLM generation",
    description=(
        "Send conversation messages to the configured Groq LLM model and return "
        "the completion with token metrics and latency. For development verification."
    ),
)
async def test_llm_generation(
    request: Request,
    payload: LLMTestRequest,
) -> SuccessResponse:
    """
    Generate text using Groq LLM for testing and development.
    """
    groq_service = getattr(request.app.state, "groq_service", None)
    if groq_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq LLM service is not available on this server.",
        )

    try:
        result = groq_service.generate(
            messages=payload.messages,
            temperature=payload.temperature,
            max_completion_tokens=payload.max_completion_tokens,
            model=payload.model,
        )

        return SuccessResponse(
            message="LLM response generated successfully.",
            data=result,
        )

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
