"""
Document data models for the extraction pipeline.

These Pydantic models define the canonical representation of extracted
documents and their pages/slides.  Every downstream stage (cleaning,
chunking, embedding, vector storage) consumes these models — they are
the single contract between document ingestion and the rest of the RAG
pipeline.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FileType(str, Enum):
    """Supported document file types."""
    PDF = "pdf"
    PPTX = "pptx"
    DOCX = "docx"
    TXT = "txt"


class ProcessingStatus(str, Enum):
    """Status of a document through the ingestion pipeline."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    CLEANING = "cleaning"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Extracted Page / Slide / Section
# ---------------------------------------------------------------------------

class ExtractedPage(BaseModel):
    """
    A single logical page, slide, or section extracted from a document.

    For PDFs this maps to a physical page, for PPTX to a slide, for DOCX
    to the overall document content (page_number may be None), and for TXT
    to the full text body.
    """
    page_number: Optional[int] = Field(
        default=None,
        description="1-indexed page number (PDF) or None for formats without pages.",
    )
    slide_number: Optional[int] = Field(
        default=None,
        description="1-indexed slide number (PPTX) or None for other formats.",
    )
    slide_title: Optional[str] = Field(
        default=None,
        description="Slide title if available (PPTX).",
    )
    heading: Optional[str] = Field(
        default=None,
        description="Section heading if detected (DOCX headings, etc.).",
    )
    text: str = Field(
        ...,
        description="The raw extracted text for this page/slide/section.",
    )
    char_count: int = Field(
        default=0,
        description="Number of characters in `text`.",
    )


# ---------------------------------------------------------------------------
# Extracted Document  (output of the extraction stage)
# ---------------------------------------------------------------------------

class ExtractedDocument(BaseModel):
    """
    Complete extraction result for a single uploaded document.

    This is what every loader returns and what the downstream pipeline
    (cleaning → chunking → embedding → Astra DB) receives.
    """
    # --- Identity ---
    document_id: str = Field(
        ...,
        description="Unique identifier for this document (supplied by the caller).",
    )
    document_name: str = Field(
        ...,
        description="Original filename of the uploaded document.",
    )
    file_type: FileType = Field(
        ...,
        description="Detected/validated file type.",
    )

    # --- Organisation ---
    user_id: str = Field(
        ...,
        description="Owning user's ID — required for data isolation.",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Subject label (e.g. 'Operating Systems').",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Topic label (e.g. 'Deadlocks').",
    )

    # --- Content ---
    pages: list[ExtractedPage] = Field(
        default_factory=list,
        description="Ordered list of extracted pages/slides/sections.",
    )
    total_pages: int = Field(
        default=0,
        description="Total number of pages/slides/sections extracted.",
    )
    total_characters: int = Field(
        default=0,
        description="Sum of characters across all pages.",
    )

    # --- Pipeline status ---
    status: ProcessingStatus = Field(
        default=ProcessingStatus.EXTRACTED,
        description="Current pipeline stage.",
    )

    # --- Timestamps ---
    created_at: datetime = Field(default_factory=datetime.utcnow)
    extraction_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken for extraction in milliseconds.",
    )


# ---------------------------------------------------------------------------
# Request / Response schemas for the document processing endpoint
# ---------------------------------------------------------------------------

class DocumentProcessRequest(BaseModel):
    """
    Request body for POST /api/v1/documents/process.

    The file itself is sent as multipart form-data; these fields are
    sent alongside the file or as query parameters.
    """
    user_id: str = Field(..., description="Owning user ID.")
    document_id: Optional[str] = Field(
        default=None,
        description="Optional pre-generated document ID. One will be created if omitted.",
    )
    subject: Optional[str] = Field(default=None, description="Subject label.")
    topic: Optional[str] = Field(default=None, description="Topic label.")


class DocumentProcessResponse(BaseModel):
    """Summary returned after successful document extraction."""
    document_id: str
    document_name: str
    file_type: FileType
    total_pages: int
    total_characters: int
    status: ProcessingStatus
    extraction_time_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Supported extensions mapping
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".pptx": FileType.PPTX,
    ".ppt": FileType.PPTX,
    ".docx": FileType.DOCX,
    ".doc": FileType.DOCX,
    ".txt": FileType.TXT,
}

SUPPORTED_MIME_TYPES: dict[str, FileType] = {
    "application/pdf": FileType.PDF,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": FileType.PPTX,
    "application/vnd.ms-powerpoint": FileType.PPTX,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCX,
    "application/msword": FileType.DOCX,
    "text/plain": FileType.TXT,
}
