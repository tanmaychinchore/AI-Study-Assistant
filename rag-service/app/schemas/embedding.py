"""
Embedding data models.

Defines the representation of embedded document chunks (chunk + vector)
and request/response schemas for the embedding test endpoint.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.chunk import DocumentChunk
from app.schemas.document import FileType, ProcessingStatus


# ---------------------------------------------------------------------------
# Embedded Chunk
# ---------------------------------------------------------------------------

class EmbeddedDocumentChunk(BaseModel):
    """
    A document chunk paired with its 1024-dimensional embedding vector.

    This is the atomic unit that gets stored in Astra DB:
      chunk metadata + text + embedding vector
    """
    # --- Chunk identity (carried forward from DocumentChunk) ---
    chunk_id: str
    chunk_index: int
    text: str
    char_count: int

    # --- Embedding ---
    embedding: list[float] = Field(
        ...,
        description="1024-dimensional embedding vector from BGE-M3.",
    )

    # --- Source metadata ---
    document_id: str
    document_name: str
    file_type: FileType
    user_id: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    slide_title: Optional[str] = None
    heading: Optional[str] = None
    source_type: str = "document"

    @classmethod
    def from_chunk_and_vector(
        cls,
        chunk: DocumentChunk,
        embedding: list[float],
    ) -> "EmbeddedDocumentChunk":
        """Create an EmbeddedDocumentChunk from a DocumentChunk + vector."""
        return cls(
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            char_count=chunk.char_count,
            embedding=embedding,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            file_type=chunk.file_type,
            user_id=chunk.user_id,
            subject=chunk.subject,
            topic=chunk.topic,
            page_number=chunk.page_number,
            slide_number=chunk.slide_number,
            slide_title=chunk.slide_title,
            heading=chunk.heading,
            source_type=chunk.source_type,
        )


# ---------------------------------------------------------------------------
# Embedded Document (collection of embedded chunks)
# ---------------------------------------------------------------------------

class EmbeddedDocument(BaseModel):
    """Result of embedding all chunks of a document."""
    document_id: str
    document_name: str
    file_type: FileType
    user_id: str
    subject: Optional[str] = None
    topic: Optional[str] = None

    embedded_chunks: list[EmbeddedDocumentChunk] = Field(default_factory=list)
    total_chunks: int = 0
    embedding_dimension: int = 1024

    # --- Timing ---
    extraction_time_ms: Optional[float] = None
    cleaning_time_ms: Optional[float] = None
    chunking_time_ms: Optional[float] = None
    embedding_time_ms: Optional[float] = None
    total_processing_time_ms: Optional[float] = None

    status: ProcessingStatus = Field(default=ProcessingStatus.EMBEDDING)


# ---------------------------------------------------------------------------
# API schemas for the embedding test endpoint
# ---------------------------------------------------------------------------

class EmbeddingTestRequest(BaseModel):
    """Request body for POST /api/v1/embeddings/test."""
    texts: list[str] = Field(
        ...,
        min_length=1,
        description="Texts to embed (1 or more).",
    )
    include_similarity: bool = Field(
        default=False,
        description="If true, include a cosine similarity matrix between all texts.",
    )


class EmbeddingTestResponse(BaseModel):
    """Response from the embedding test endpoint."""
    model: str
    device: str
    embedding_dimension: int
    count: int
    texts: list[str]
    embedding_time_ms: float
    similarity_matrix: Optional[list[list[float]]] = Field(
        default=None,
        description="Cosine similarity between each pair of texts (if requested).",
    )
