from fastapi import APIRouter, Request

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.response import HealthCheckData, SuccessResponse

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=SuccessResponse,
    summary="Service health check",
    description="Returns the current health status of the RAG service and its sub-components.",
)
async def health_check(request: Request) -> SuccessResponse:
    """Return service health status with version, environment, and sub-component info."""
    logger.info("Health check requested")

    components: dict[str, str] = {}

    # Check Embedding service
    embedding_service = getattr(request.app.state, "embedding_service", None)
    if embedding_service and embedding_service.is_loaded:
        components["embedding_model"] = f"loaded ({embedding_service.model_name}, device={embedding_service.device})"
    else:
        components["embedding_model"] = "unavailable"

    # Check Astra DB service
    astra_service = getattr(request.app.state, "astra_db_service", None)
    if astra_service and astra_service.is_ready:
        components["astra_db"] = f"connected ({astra_service.collection_name}, dim={astra_service.expected_dimension})"
    elif astra_service and not astra_service.is_configured:
        components["astra_db"] = "not_configured"
    else:
        components["astra_db"] = "unavailable"

    health_data = HealthCheckData(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        components=components,
    )

    return SuccessResponse(
        message="RAG service is running.",
        data=health_data.model_dump(),
    )

