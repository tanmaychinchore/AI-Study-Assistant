"""
Authentication and User Pydantic schemas.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    """Schema for user registration requests."""
    email: EmailStr = Field(..., description="Unique user email address.")
    password: str = Field(..., min_length=8, description="User password (minimum 8 characters).")
    name: Optional[str] = Field(None, description="Optional user display name.")


class LoginRequest(BaseModel):
    """Schema for user login requests."""
    email: EmailStr = Field(..., description="User email address.")
    password: str = Field(..., description="User password.")


class UserResponse(BaseModel):
    """Schema for user resource responses."""
    id: str = Field(..., alias="_id", description="Unique user ID.")
    email: EmailStr = Field(..., description="User email address.")
    name: Optional[str] = Field(None, description="User display name.")
    created_at: datetime = Field(..., description="Account creation timestamp.")

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class TokenResponse(BaseModel):
    """Schema for successful authentication token responses."""
    access_token: str = Field(..., description="JWT access token.")
    token_type: str = Field("bearer", description="Token type prefix.")
    expires_in: int = Field(..., description="Token lifespan in seconds.")
    user: UserResponse = Field(..., description="Authenticated user info.")
