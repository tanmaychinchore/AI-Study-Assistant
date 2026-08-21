"""
Smart Tutor API routes.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies.auth import get_current_user
from app.schemas.tutor import TutorRequest, TutorContextResponse
from app.schemas.response import ErrorResponse
from app.services.tutor_service import TutorService

router = APIRouter(prefix="/tutor", tags=["Smart Tutor"])


def _get_tutor_service(request: Request) -> TutorService:
    """Retrieve TutorService from application state."""
    service = getattr(request.app.state, "tutor_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Smart Tutor service is not available on this server.",
        )
    return service


@router.post(
    "/context",
    response_model=TutorContextResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid tutor request"},
        401: {"model": ErrorResponse, "description": "Unauthorized access"},
        503: {"model": ErrorResponse, "description": "Tutor service starting up"},
    },
    summary="Retrieve user study context for Smart Tutor features",
    description="Returns semantic chunks and citation counts scoped strictly to the authenticated user.",
)
async def get_tutor_context(
    request_body: TutorRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> TutorContextResponse:
    """Retrieve user study context grounded strictly by tenant isolation."""
    tutor_service = _get_tutor_service(request)
    try:
        context_data = tutor_service.get_study_context(
            request=request_body,
            user_id=current_user["user_id"],
        )
        return TutorContextResponse(
            success=True,
            message="Study context retrieved successfully.",
            data=context_data,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tutor context retrieval failed: {exc}",
        ) from exc
