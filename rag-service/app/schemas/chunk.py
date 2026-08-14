"""
Chunk data models for the chunking pipeline.

A chunk is a small, semantically meaningful piece of text extracted
from a document page/slide/section.  Chunks carry forward all the
metadata from their source page so the RAG pipeline can later provide
accurate source citations.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.document import FileType, ProcessingStatus


# ---------------------------------------------------------------------------
# Single Chunk
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """
    A single chunk of text ready for embedding and vector storage.

    This is the atomic unit that flows into:
      embedding → Astra DB → retrieval → LLM context
    """
    # --- Chunk identity ---
    chunk_id: str = Field(
        ...,
        description="Unique identifier for this chunk (e.g. '{doc_id}_chunk_001').",
    )
    chunk_index: int = Field(
        ...,
        description="0-based position of this chunk in the document.",
    )

    # --- Content ---
    text: str = Field(
        ...,
        description="The chunk text content.",
    )
    char_count: int = Field(
        default=0,
        description="Number of characters in `text`.",
    )

    # --- Source metadata (inherited from ExtractedPage) ---
    document_id: str = Field(..., description="Parent document ID.")
    document_name: str = Field(..., description="Original filename.")
    file_type: FileType = Field(..., description="Source file type.")
    user_id: str = Field(..., description="Owning user ID.")
    subject: Optional[str] = Field(default=None, description="Subject label.")
    topic: Optional[str] = Field(default=None, description="Topic label.")

    # --- Page/slide source ---
    page_number: Optional[int] = Field(
        default=None,
        description="Source page number (PDF).",
    )
    slide_number: Optional[int] = Field(
        default=None,
        description="Source slide number (PPTX).",
    )
    slide_title: Optional[str] = Field(
        default=None,
        description="Source slide title (PPTX).",
    )
    heading: Optional[str] = Field(
        default=None,
        description="Source section heading (DOCX).",
    )

    source_type: str = Field(
        default="document",
        description="Type of source (always 'document' for now).",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Chunked Document (output of the chunking stage)
# ---------------------------------------------------------------------------

class ChunkedDocument(BaseModel):
    """
    Complete chunking result for a single document.

    This is what the chunking service returns and what the
    embedding → vector storage pipeline receives.
    """
    document_id: str
    document_name: str
    file_type: FileType
    user_id: str
    subject: Optional[str] = None
    topic: Optional[str] = None

    chunks: list[DocumentChunk] = Field(
        default_factory=list,
        description="Ordered list of document chunks.",
    )
    total_chunks: int = Field(default=0, description="Number of chunks.")
    total_characters: int = Field(default=0, description="Sum of all chunk characters.")

    # --- Configuration used ---
    chunk_size: int = Field(..., description="Chunk size setting used.")
    chunk_overlap: int = Field(..., description="Chunk overlap setting used.")

    # --- Pipeline status ---
    status: ProcessingStatus = Field(default=ProcessingStatus.CHUNKING)

    # --- Timing ---
    extraction_time_ms: Optional[float] = None
    cleaning_time_ms: Optional[float] = None
    chunking_time_ms: Optional[float] = None
    total_processing_time_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# API Response for the full process+chunk endpoint
# ---------------------------------------------------------------------------

class DocumentChunkPreview(BaseModel):
    """A lightweight chunk preview for API responses (no full text)."""
    chunk_id: str
    chunk_index: int
    char_count: int
    text_preview: str = Field(
        ...,
        description="First 200 characters of the chunk text.",
    )
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    slide_title: Optional[str] = None
    heading: Optional[str] = None


class DocumentProcessAndChunkResponse(BaseModel):
    """Response from the full extract → clean → chunk pipeline."""
    document_id: str
    document_name: str
    file_type: FileType
    user_id: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    total_pages: int
    total_chunks: int
    total_characters: int
    chunk_size: int
    chunk_overlap: int
    status: ProcessingStatus
    extraction_time_ms: Optional[float] = None
    cleaning_time_ms: Optional[float] = None
    chunking_time_ms: Optional[float] = None
    total_processing_time_ms: Optional[float] = None
    chunks_preview: list[DocumentChunkPreview] = Field(
        default_factory=list,
        description="Preview of first few chunks.",
    )
