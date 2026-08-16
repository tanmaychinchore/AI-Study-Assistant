"""
Indexing pipeline schemas.

Defines the data models for the complete end-to-end indexing pipeline:
Extraction → Cleaning → Chunking → BGE-M3 Embedding → Astra DB Storage.
"""

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.chunk import DocumentChunkPreview
from app.schemas.document import FileType, ProcessingStatus


class IndexingStatistics(BaseModel):
    """Timing breakdown for every stage in the indexing pipeline."""
    extraction_time_ms: float = Field(..., description="Time taken to extract text and structure.")
    cleaning_time_ms: float = Field(..., description="Time taken to clean and normalize text.")
    chunking_time_ms: float = Field(..., description="Time taken to split text into chunks.")
    embedding_time_ms: float = Field(..., description="Time taken to generate 1024-dim BGE-M3 embeddings.")
    astra_insertion_time_ms: float = Field(..., description="Time taken to store vectors and metadata in Astra DB.")
    total_time_ms: float = Field(..., description="Total end-to-end indexing duration.")


class IndexingResult(BaseModel):
    """Complete response payload for a successfully indexed document."""
    document_id: str = Field(..., description="Unique identifier for the document.")
    document_name: str = Field(..., description="Original file name.")
    file_type: FileType = Field(..., description="Detected document format.")
    user_id: str = Field(..., description="Owning user ID.")
    subject: Optional[str] = Field(default=None, description="Subject category.")
    topic: Optional[str] = Field(default=None, description="Topic category.")
    total_pages: int = Field(..., description="Total pages or slides in the source document.")
    total_chunks: int = Field(..., description="Total number of chunks produced.")
    total_characters: int = Field(..., description="Total character count across all chunks.")
    embeddings_generated: int = Field(..., description="Number of 1024-dim vector embeddings generated.")
    vectors_inserted: int = Field(..., description="Number of vector documents persisted in Astra DB.")
    collection: str = Field(..., description="Astra DB collection name where vectors are stored.")
    status: ProcessingStatus = Field(default=ProcessingStatus.INDEXED, description="Final indexing status.")
    statistics: IndexingStatistics = Field(..., description="Stage-by-stage timing metrics.")
    chunks_preview: list[DocumentChunkPreview] = Field(default_factory=list, description="Preview of first few indexed chunks.")
