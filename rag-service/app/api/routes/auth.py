"""
Authentication endpoints for user registration and login.
"""

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.auth import (
    LoginRequest,
    TokenResponse,
    UserRegisterRequest,
    UserResponse,
)
from app.schemas.response import ErrorResponse, SuccessResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _get_auth_service(request: Request) -> AuthService:
    """Retrieve AuthService from application state."""
    service = getattr(request.app.state, "auth_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is starting up.",
        )
    return service


@router.post(
    "/register",
    response_model=SuccessResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Registration failed"},
    },
    summary="Register a new user",
    description="Creates a new normalized user account in the system.",
)
async def register(
    request_body: UserRegisterRequest,
    request: Request,
) -> SuccessResponse:
    auth_service = _get_auth_service(request)
    try:
        user_doc = auth_service.register_user(
            email=request_body.email,
            password=request_body.password,
            name=request_body.name,
        )
        user_response = UserResponse(
            _id=user_doc["user_id"],
            email=user_doc["email"],
            name=user_doc.get("name"),
            created_at=user_doc["created_at"],
        )
        return SuccessResponse(
            message="User registration successful.",
            data=user_response.model_dump(by_alias=True),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid credentials"},
    },
    summary="User login",
    description="Authenticates credentials and returns a JWT access token.",
)
async def login(
    request_body: LoginRequest,
    request: Request,
) -> TokenResponse:
    auth_service = _get_auth_service(request)
    try:
        user_doc = auth_service.authenticate_user(
            email=request_body.email,
            password=request_body.password,
        )
        token, expires_in = auth_service.create_access_token(user_doc["user_id"])
        
        user_response = UserResponse(
            _id=user_doc["user_id"],
            email=user_doc["email"],
            name=user_doc.get("name"),
            created_at=user_doc["created_at"],
        )
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=expires_in,
            user=user_response,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
