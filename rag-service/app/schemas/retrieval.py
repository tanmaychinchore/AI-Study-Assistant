"""
Retrieval data models.

Defines schemas for:
  - RetrievalRequest     — Query, user_id, top_k, optional filters & similarity threshold
  - RetrievedChunk       — Document chunk augmented with similarity score and full metadata
  - RetrievalStatistics  — Execution time breakdown and chunk counts
  - RetrievalResult      — Complete response payload for retrieval operations
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class RetrievalRequest(BaseModel):
    """
    Request model for semantic retrieval search.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question or search query.",
        examples=["What is a Process Control Block in operating systems?"],
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description="ID of the user issuing the query (enforces strict user isolation).",
        examples=["student_demo_101"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant chunks to retrieve (1–50, default: 5).",
        examples=[5],
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional filter: restrict search to chunks of a specific document.",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Optional filter: restrict search to chunks of a specific subject.",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Optional filter: restrict search to chunks of a specific topic.",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity cutoff score (0.0–1.0).",
    )

    @field_validator("query", "user_id")
    @classmethod
    def validate_non_empty_strings(cls, value: str, info) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace-only.")
        return stripped


class RetrievedChunk(BaseModel):
    """
    A single retrieved chunk ranked by semantic similarity score.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier in Astra DB.")
    document_id: str = Field(..., description="ID of the parent document.")
    document_name: str = Field(..., description="Original filename of the parent document.")
    user_id: str = Field(..., description="Owner user ID.")
    text: str = Field(..., description="Extracted & cleaned text content.")
    similarity_score: float = Field(
        ...,
        description="Cosine similarity score (0.0–1.0, higher is more relevant).",
    )
    char_count: int = Field(..., description="Character count of the chunk text.")
    file_type: str = Field(..., description="Source file format (pdf, pptx, docx, txt).")
    page_number: Optional[int] = Field(None, description="1-indexed page number if PDF/DOCX.")
    slide_number: Optional[int] = Field(None, description="1-indexed slide number if PPTX.")
    slide_title: Optional[str] = Field(None, description="Slide title if PPTX.")
    heading: Optional[str] = Field(None, description="Section heading if DOCX/PDF.")
    subject: Optional[str] = Field(None, description="Subject tag if provided.")
    topic: Optional[str] = Field(None, description="Topic tag if provided.")
    chunk_index: int = Field(..., description="0-indexed position within the document.")
    source_type: str = Field(default="document", description="Origin of the chunk.")


class RetrievalStatistics(BaseModel):
    """
    Performance timings and count statistics for the retrieval operation.
    """
    embedding_time_ms: float = Field(..., description="Time taken to embed query with BGE-M3.")
    search_time_ms: float = Field(..., description="Time taken for Astra DB vector similarity query.")
    total_time_ms: float = Field(..., description="Total end-to-end retrieval time.")
    chunks_retrieved: int = Field(..., description="Number of candidate chunks fetched from Astra DB.")
    chunks_returned: int = Field(..., description="Number of chunks returned after filtering.")


class RetrievalResult(BaseModel):
    """
    Complete result payload returned by RetrievalService and API endpoint.
    """
    query: str = Field(..., description="The original search query.")
    user_id: str = Field(..., description="User ID for which retrieval was executed.")
    top_k: int = Field(..., description="Requested top-K count.")
    filters_applied: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of filters applied during retrieval.",
    )
    results: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Ranked list of relevant chunks ordered by descending similarity score.",
    )
    statistics: RetrievalStatistics = Field(
        ...,
        description="Timing breakdown and chunk counts.",
    )
