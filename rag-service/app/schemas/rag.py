"""
Pydantic schemas for the complete RAG generation pipeline.

Defines schemas for RAG query requests, source citations, pipeline statistics,
and end-to-end grounded generation results.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


class RAGRequest(BaseModel):
    """Request payload for the end-to-end RAG question answering endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language study question from the student.",
        examples=["What are the four necessary conditions for deadlock?"],
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique student/user identifier to enforce data privacy and isolation.",
        examples=["student_alice"],
    )
    top_k: int = Field(
        default=settings.TOP_K,
        ge=settings.MIN_TOP_K,
        le=settings.MAX_TOP_K,
        description=f"Maximum candidate chunks to retrieve ({settings.MIN_TOP_K}–{settings.MAX_TOP_K}).",
        examples=[5],
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a specific document ID.",
        examples=[None],
    )
    subject: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a specific academic subject.",
        examples=[None],
    )
    topic: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a specific topic area.",
        examples=[None],
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity cutoff (0.0–1.0). Chunks below this score are dropped.",
        examples=[None],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the four necessary conditions for deadlock?",
                "user_id": "student_alice",
                "top_k": 5,
                "document_id": None,
                "subject": None,
                "topic": None,
                "similarity_threshold": None,
            }
        }
    }

    @field_validator("query")
    @classmethod
    def validate_query_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")
        return v.strip()

    @field_validator("user_id")
    @classmethod
    def validate_user_id_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("User ID cannot be empty or whitespace-only.")
        return v.strip()


class RAGSource(BaseModel):
    """Source citation metadata for a document chunk used in the grounded context."""

    source_id: str = Field(
        ...,
        description="Formatted source identifier (e.g. '[SOURCE 1]').",
        examples=["[SOURCE 1]"],
    )
    chunk_id: str = Field(
        ...,
        description="Unique identifier of the stored chunk.",
    )
    document_id: str = Field(
        ...,
        description="Identifier of the parent document.",
    )
    document_name: str = Field(
        ...,
        description="Original filename of the study material.",
        examples=["Operating_Systems_Chapter6.pdf"],
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number (1-indexed) if source is a PDF or document.",
    )
    slide_number: Optional[int] = Field(
        default=None,
        description="Slide number (1-indexed) if source is a presentation.",
    )
    slide_title: Optional[str] = Field(
        default=None,
        description="Slide title if available.",
    )
    heading: Optional[str] = Field(
        default=None,
        description="Nearest section heading if available.",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Academic subject category.",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Specific study topic.",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score of this chunk against the query.",
    )


class RAGRetrievalStatistics(BaseModel):
    """Retrieval stage performance and count metrics."""

    chunks_retrieved: int = Field(
        ...,
        ge=0,
        description="Total candidate chunks retrieved by semantic search.",
    )
    chunks_used_as_context: int = Field(
        ...,
        ge=0,
        description="Number of chunks that fit within the context budget and were passed to LLM.",
    )
    retrieval_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time in milliseconds spent on embedding and vector search.",
    )


class RAGGenerationStatistics(BaseModel):
    """LLM generation stage performance and token metrics."""

    model: str = Field(
        ...,
        description="Exact model identifier used for generating the response.",
    )
    input_tokens: int = Field(
        default=0,
        ge=0,
        description="Prompt tokens consumed by the LLM.",
    )
    output_tokens: int = Field(
        default=0,
        ge=0,
        description="Completion tokens produced by the LLM.",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total token consumption (input + output).",
    )
    generation_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time in milliseconds spent generating the completion.",
    )
    finish_reason: str = Field(
        default="stop",
        description="Reason generation terminated ('stop', 'length', etc.).",
    )


class RAGResult(BaseModel):
    """Complete end-to-end grounded question answering result."""

    query: str = Field(
        ...,
        description="The original user question.",
    )
    user_id: str = Field(
        ...,
        description="The user ID for whom the query was executed.",
    )
    answer: str = Field(
        ...,
        description="Grounded explanation generated by the assistant.",
    )
    grounded: bool = Field(
        ...,
        description="True if the answer was generated from retrieved study material, False if no context was found.",
    )
    sources: list[RAGSource] = Field(
        default_factory=list,
        description="List of cited source chunks used in the grounded answer.",
    )
    retrieval_statistics: RAGRetrievalStatistics = Field(
        ...,
        description="Detailed timing and counts for the retrieval stage.",
    )
    generation_statistics: RAGGenerationStatistics = Field(
        ...,
        description="Detailed timing and token usage for the generation stage.",
    )
    context_building_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time in milliseconds spent formatting and budgeting context chunks.",
    )
    total_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total end-to-end pipeline latency in milliseconds.",
    )
