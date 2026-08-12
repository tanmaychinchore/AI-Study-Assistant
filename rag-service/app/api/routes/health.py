"""
Health check route.

Provides a lightweight endpoint for liveness probes and
integration testing. Will later include sub-component checks
(Astra DB connectivity, embedding model status, Groq reachability).
"""

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.response import HealthCheckData, SuccessResponse

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=SuccessResponse,
    summary="Service health check",
    description="Returns the current health status of the RAG service.",
)
async def health_check() -> SuccessResponse:
    """Return service health status with version and environment info."""
    logger.info("Health check requested")

    health_data = HealthCheckData(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    return SuccessResponse(
        message="RAG service is running.",
        data=health_data.model_dump(),
    )
