"""
Pydantic schemas for Groq LLM service.

Defines schemas for chat messages, LLM test requests, structured generation results,
and service health/readiness reporting.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="The role of the message author: 'system', 'user', or 'assistant'.",
        examples=["user"],
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The textual content of the message.",
        examples=["Explain deadlock in simple terms."],
    )

    @field_validator("content")
    @classmethod
    def validate_content_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty or whitespace-only.")
        return v.strip()


class LLMTestRequest(BaseModel):
    """Request payload for the development/testing LLM endpoint."""

    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="List of conversation messages in chronological order.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model identifier override. Defaults to configured GROQ_MODEL.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0.0 and 2.0. Defaults to configured value (0.2).",
    )
    max_completion_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate. Defaults to configured value (1024).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful educational assistant."
                    },
                    {
                        "role": "user",
                        "content": "Explain deadlock in simple terms."
                    }
                ]
            }
        }
    }

    @field_validator("messages")
    @classmethod
    def validate_messages_list(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("At least one message is required.")
        return v


class GenerationResult(BaseModel):
    """
    Standardized response schema returned by the Groq LLM service.

    Transforms raw provider responses into our application schema with token usage,
    latency metrics, model identifier, and completion metadata.
    """

    content: str = Field(
        ...,
        description="Generated text response from the language model.",
    )
    model: str = Field(
        ...,
        description="The exact model identifier used for generation.",
    )
    finish_reason: str = Field(
        default="stop",
        description="Reason generation terminated ('stop', 'length', 'tool_calls', etc.).",
    )
    input_tokens: int = Field(
        ...,
        ge=0,
        description="Number of prompt/input tokens consumed.",
    )
    output_tokens: int = Field(
        ...,
        ge=0,
        description="Number of completion/output tokens generated.",
    )
    total_tokens: int = Field(
        ...,
        ge=0,
        description="Total tokens consumed (input + output).",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Total time in milliseconds taken for the API request.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request ID returned by Groq for debugging/tracing.",
    )


class LLMHealthInfo(BaseModel):
    """Readiness and configuration report for the Groq LLM service."""

    status: Literal["ready", "not_configured", "unavailable"] = Field(
        ...,
        description="Service readiness status.",
    )
    provider: str = Field(
        default="groq",
        description="LLM provider name.",
    )
    model: str = Field(
        ...,
        description="Configured default model identifier.",
    )
    configured: bool = Field(
        ...,
        description="Whether a valid API key is present in configuration.",
    )
