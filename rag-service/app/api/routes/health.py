from fastapi import APIRouter, Request, Response, status

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.response import HealthCheckData, SuccessResponse, ErrorResponse

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
    logger.debug("Health check requested")

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

    # Check Groq LLM service (Task 8)
    groq_service = getattr(request.app.state, "groq_service", None)
    if groq_service and groq_service.is_ready:
        components["llm"] = f"ready (groq/{groq_service.model})"
    elif groq_service and not groq_service.is_configured:
        components["llm"] = "not_configured"
    else:
        components["llm"] = "unavailable"

    # Check MongoDB Conversation store status (Task 10/12)
    conversation_service = getattr(request.app.state, "conversation_service", None)
    if conversation_service and conversation_service._client:
        try:
            # lightweight ping to verify connection
            conversation_service._client.admin.command('ping')
            components["mongodb"] = "ready"
        except Exception:
            components["mongodb"] = "unavailable"
    else:
        components["mongodb"] = "unavailable"

    # Determine overall status
    is_healthy = all(v != "unavailable" for v in components.values())
    overall_status = "healthy" if is_healthy else "degraded"

    health_data = HealthCheckData(
        status=overall_status,
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        components=components,
    )

    return SuccessResponse(
        message=f"RAG service status is {overall_status}.",
        data=health_data.model_dump(),
    )


@router.get(
    "/health/liveness",
    response_model=SuccessResponse,
    summary="Liveness check",
    description="Returns HTTP 200 if the service process is active.",
)
async def liveness_check() -> SuccessResponse:
    """Basic liveness probe returning HTTP 200 if process is up."""
    return SuccessResponse(
        message="Service is alive.",
        data={"status": "alive"}
    )


@router.get(
    "/health/readiness",
    responses={
        200: {"model": SuccessResponse, "description": "Ready to serve traffic"},
        503: {"model": ErrorResponse, "description": "Sub-components down"}
    },
    summary="Readiness check",
    description="Verifies that all required sub-components are fully operational.",
)
async def readiness_check(request: Request, response: Response):
    """Readiness probe checking critical component connections. Returns 503 if any are down."""
    health_resp = await health_check(request)
    components = health_resp.data["components"]

    # Critical components required to serve traffic
    critical_failures = []
    if components.get("embedding_model") == "unavailable":
        critical_failures.append("embedding_model")
    if components.get("astra_db") == "unavailable":
        critical_failures.append("astra_db")
    if components.get("mongodb") == "unavailable":
        critical_failures.append("mongodb")
    if components.get("llm") == "unavailable":
        critical_failures.append("llm")

    if critical_failures:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ErrorResponse(
            success=False,
            message=f"Service not ready. Sub-component failures: {', '.join(critical_failures)}",
            data={"components": components}
        )

    return health_resp

