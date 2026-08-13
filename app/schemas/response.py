"""
Pydantic response schemas for consistent API responses.

Every endpoint returns data wrapped in one of these schemas
so the Node.js backend always receives a predictable JSON shape.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Generic API Responses
# ---------------------------------------------------------------------------

class SuccessResponse(BaseModel):
    """Standard success envelope for all API responses."""
    success: bool = True
    message: str = "Request completed successfully."
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error envelope for all API error responses."""
    success: bool = False
    message: str = "An error occurred."
    error_code: Optional[str] = None
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class HealthCheckData(BaseModel):
    """Payload returned by the health check endpoint."""
    status: str = "healthy"
    service: str = ""
    version: str = ""
    environment: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Future: add sub-component health (Astra DB, embedding model, Groq)
    components: Optional[dict[str, str]] = None
