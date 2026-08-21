"""
Pydantic schemas for conversations, messages, and multi-turn chat.

Defines data models for conversation lifecycle, message history,
and conversation-aware RAG chat requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.schemas.rag import (
    RAGGenerationStatistics,
    RAGRetrievalStatistics,
    RAGSource,
)


class MessageRole(str, Enum):
    """Allowed roles for conversation messages."""
    USER = "user"
    ASSISTANT = "assistant"


# ---------------------------------------------------------------------------
# Conversation Lifecycle Schemas
# ---------------------------------------------------------------------------

class ConversationCreateRequest(BaseModel):
    """Request payload to create a new study conversation."""

    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique user/student identifier.",
        examples=["student_alice"],
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional human-friendly title for the conversation.",
        examples=["Operating Systems Revision"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "student_alice",
                "title": "Operating Systems Revision",
            }
        }
    }

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("User ID cannot be empty or whitespace-only.")
        return v.strip()

    @field_validator("title")
    @classmethod
    def sanitize_title(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            stripped = v.strip()
            return stripped if stripped else None
        return None


class ConversationResponse(BaseModel):
    """Details of a single conversation session."""

    conversation_id: str = Field(
        ...,
        description="Unique UUID identifying the conversation.",
    )
    user_id: str = Field(
        ...,
        description="User ID owning this conversation.",
    )
    title: str = Field(
        ...,
        description="Title or topic label of the conversation.",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the conversation was initiated.",
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp of the most recent message or update.",
    )
    message_count: int = Field(
        default=0,
        ge=0,
        description="Total number of messages exchanged in this conversation.",
    )


class ConversationSummary(BaseModel):
    """Lightweight conversation metadata for listing endpoints."""

    conversation_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationListResponse(BaseModel):
    """Response payload for listing a user's conversations."""

    conversations: list[ConversationResponse] = Field(
        default_factory=list,
        description="List of conversations sorted by updated_at descending.",
    )
    total: int = Field(
        default=0,
        ge=0,
        description="Total number of conversations found for the user.",
    )


# ---------------------------------------------------------------------------
# Message Schemas
# ---------------------------------------------------------------------------

class MessageResponse(BaseModel):
    """A single persistent conversation message."""

    message_id: str = Field(
        ...,
        description="Unique UUID identifying this message.",
    )
    conversation_id: str = Field(
        ...,
        description="Conversation this message belongs to.",
    )
    user_id: str = Field(
        ...,
        description="User ID owning the message.",
    )
    role: MessageRole = Field(
        ...,
        description="Author role of the message (user or assistant).",
    )
    content: str = Field(
        ...,
        description="Message content string.",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the message was recorded.",
    )


class MessageListResponse(BaseModel):
    """Response payload for retrieving messages in a conversation."""

    conversation_id: str
    messages: list[MessageResponse] = Field(
        default_factory=list,
        description="Ordered list of messages (oldest to newest).",
    )
    total: int = Field(
        default=0,
        ge=0,
        description="Total count of messages returned.",
    )


# ---------------------------------------------------------------------------
# Multi-turn Chat Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request payload for conversation-aware multi-turn RAG chat."""

    user_id: str = Field(
        ...,
        min_length=1,
        description="User ID sending the message (must match conversation owner).",
        examples=["student_alice"],
    )
    message: str = Field(
        ...,
        min_length=1,
        description="User question or follow-up query.",
        examples=["What are its states?"],
    )
    top_k: int = Field(
        default=settings.TOP_K,
        ge=settings.MIN_TOP_K,
        le=settings.MAX_TOP_K,
        description="Maximum candidate chunks to retrieve.",
        examples=[5],
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a specific document ID.",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a specific academic subject.",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a specific topic area.",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity cutoff.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "student_alice",
                "message": "What are its states?",
                "top_k": 5,
            }
        }
    }

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("User ID cannot be empty or whitespace-only.")
        return v.strip()

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message cannot be empty or whitespace-only.")
        return v.strip()


class ChatResponse(BaseModel):
    """Response payload for conversation-aware chat generation."""

    conversation_id: str = Field(
        ...,
        description="ID of the active conversation.",
    )
    user_message: MessageResponse = Field(
        ...,
        description="The recorded user message record.",
    )
    assistant_message: Optional[MessageResponse] = Field(
        default=None,
        description="The recorded assistant response record.",
    )
    answer: str = Field(
        ...,
        description="Grounded explanation generated by the assistant.",
    )
    grounded: bool = Field(
        ...,
        description="True if answered from retrieved study material, False if no context was found.",
    )
    sources: list[RAGSource] = Field(
        default_factory=list,
        description="List of cited source chunks used in the grounded answer.",
    )
    retrieval_statistics: RAGRetrievalStatistics = Field(
        ...,
        description="Performance timing and candidate count metrics for the retrieval stage.",
    )
    generation_statistics: RAGGenerationStatistics = Field(
        ...,
        description="Detailed token metrics and latency for the generation stage.",
    )
    total_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total end-to-end multi-turn chat latency in milliseconds.",
    )
