"""
Smart Tutor schemas.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievedChunk


class TutorRequest(BaseModel):
    """Schema for tutor queries and study context requests."""
    query: str = Field(..., description="Query or instruction to the tutor.")
    document_id: Optional[str] = Field(None, description="Optional document ID filter.")
    subject: Optional[str] = Field(None, description="Optional subject label.")
    topic: Optional[str] = Field(None, description="Optional topic label.")
    difficulty: Optional[str] = Field("medium", description="Learning difficulty (easy | medium | hard).")
    conversation_id: Optional[str] = Field(None, description="Optional conversation session ID.")


class TutorContextData(BaseModel):
    """Details of the retrieved study context for verification."""
    query: str = Field(..., description="The query executed.")
    chunks: list[RetrievedChunk] = Field(..., description="List of retrieved study chunks.")
    citations_count: int = Field(..., description="Number of preserved citations.")


class TutorContextResponse(BaseModel):
    """Schema for tutor context response payload."""
    success: bool = Field(True, description="Indicates request success.")
    message: str = Field(..., description="Response summary message.")
    data: TutorContextData = Field(..., description="Retrieved study context data.")
