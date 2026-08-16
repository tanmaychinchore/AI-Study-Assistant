"""
Astra DB schemas.

Defines the data models for Astra DB health checks, chunk insertion requests/responses,
and chunk retrieval payloads.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AstraDBHealthResponse(BaseModel):
    """Health information for Astra DB."""
    status: str = Field(..., description="Connection status (connected | disconnected | error | not_configured).")
    keyspace: str = Field(..., description="Target keyspace name.")
    collection: str = Field(..., description="Target vector collection name.")
    vector_dimension: int = Field(default=1024, description="Expected vector dimension.")
    metric: str = Field(default="cosine", description="Similarity metric (cosine).")
    is_connected: bool = Field(default=False, description="Whether database is reachable.")
    collection_exists: bool = Field(default=False, description="Whether the collection exists.")
    detail: Optional[str] = Field(default=None, description="Detailed diagnostic or error message.")


class AstraDBInsertRequest(BaseModel):
    """Request payload for manual/test chunk insertion into Astra DB."""
    text: str = Field(..., description="Text content of the chunk.")
    document_name: str = Field(default="test_doc.txt", description="Source document name.")
    document_id: Optional[str] = Field(default="test_doc_001", description="Source document identifier.")
    user_id: Optional[str] = Field(default="test_user", description="Owner user ID.")
    subject: Optional[str] = Field(default=None, description="Subject category.")
    topic: Optional[str] = Field(default=None, description="Topic category.")
    page_number: Optional[int] = Field(default=None, description="Page number if applicable.")
    slide_number: Optional[int] = Field(default=None, description="Slide number if applicable.")
    slide_title: Optional[str] = Field(default=None, description="Slide title if applicable.")
    heading: Optional[str] = Field(default=None, description="Section heading if applicable.")


class AstraDBInsertResponse(BaseModel):
    """Response returned after inserting chunks into Astra DB."""
    inserted_count: int = Field(..., description="Number of chunks successfully inserted.")
    failed_count: int = Field(default=0, description="Number of chunks that failed insertion.")
    inserted_ids: list[str] = Field(default_factory=list, description="IDs of inserted chunks.")
    duration_ms: float = Field(..., description="Time taken for insertion in milliseconds.")
    collection: str = Field(..., description="Collection where records were stored.")


class AstraDBChunkResponse(BaseModel):
    """Response representation of a single chunk retrieved from Astra DB."""
    chunk_id: str
    document_id: str
    document_name: str
    user_id: str
    text: str
    char_count: int
    file_type: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    slide_title: Optional[str] = None
    heading: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    chunk_index: int
    source_type: str
    has_vector: bool = True
    vector_dimension: int = 1024
    created_at: Optional[str] = None
