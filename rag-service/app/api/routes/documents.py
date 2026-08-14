"""
Document processing API routes.

Provides endpoints for:
  - /process         — extract only (Task 2)
  - /process-chunks  — extract → clean → chunk (Task 3)
"""

import tempfile
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.logging import get_logger
from app.schemas.document import (
    DocumentProcessResponse,
    FileType,
    SUPPORTED_EXTENSIONS,
)
from app.schemas.chunk import (
    DocumentChunkPreview,
    DocumentProcessAndChunkResponse,
)
from app.schemas.response import ErrorResponse, SuccessResponse
from app.services.extraction_service import extract_document, get_file_type
from app.services.document_service import process_document_pipeline

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# Maximum file size: 50 MB
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Number of chunk previews to include in the response
CHUNK_PREVIEW_COUNT = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _validate_and_read_file(file: UploadFile) -> tuple[bytes, str]:
    """Validate the upload and return (contents, extension)."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{file_ext}'. "
                f"Supported formats: {supported}"
            ),
        )

    try:
        contents = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size ({len(contents) / (1024 * 1024):.1f} MB) exceeds "
                f"maximum allowed size ({MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB)."
            ),
        )

    return contents, file_ext


# ---------------------------------------------------------------------------
# POST /documents/process — extraction only
# ---------------------------------------------------------------------------

@router.post(
    "/process",
    response_model=SuccessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or request"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Extraction failed"},
    },
    summary="Upload and extract a document",
    description=(
        "Upload a PDF, PPTX, DOCX, or TXT file.  The service extracts text "
        "and metadata, returning a structured extraction result.  This is the "
        "first step of the document ingestion pipeline."
    ),
)
async def process_document(
    file: UploadFile = File(..., description="The document file to process."),
    user_id: str = Form(..., description="Owning user's ID."),
    document_id: str = Form(default=None, description="Optional pre-generated document ID."),
    subject: str = Form(default=None, description="Subject label."),
    topic: str = Form(default=None, description="Topic label."),
) -> SuccessResponse:
    """Upload a document and extract its text and metadata."""
    contents, _ = await _validate_and_read_file(file)

    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="rag_extract_")
        tmp_path = Path(tmp_dir) / file.filename
        tmp_path.write_bytes(contents)

        logger.info(
            "Processing document: name=%s  size=%.1fKB  user=%s",
            file.filename,
            len(contents) / 1024,
            user_id,
        )

        extracted = extract_document(
            file_path=tmp_path,
            user_id=user_id,
            document_id=document_id or None,
            subject=subject or None,
            topic=topic or None,
        )

        response_data = DocumentProcessResponse(
            document_id=extracted.document_id,
            document_name=extracted.document_name,
            file_type=extracted.file_type,
            total_pages=extracted.total_pages,
            total_characters=extracted.total_characters,
            status=extracted.status,
            extraction_time_ms=extracted.extraction_time_ms,
        )

        return SuccessResponse(
            message=f"Document '{file.filename}' extracted successfully.",
            data=response_data.model_dump(),
        )

    except ValueError as exc:
        logger.warning("Validation error during extraction: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except FileNotFoundError as exc:
        logger.error("File not found during extraction: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document extraction failed: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during document processing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during document processing.",
        )
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# POST /documents/process-chunks — full extract → clean → chunk pipeline
# ---------------------------------------------------------------------------

@router.post(
    "/process-chunks",
    response_model=SuccessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or request"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Processing failed"},
    },
    summary="Upload, extract, clean, and chunk a document",
    description=(
        "Full pipeline: Upload a document → extract text → clean text → "
        "split into chunks.  Returns chunk statistics and a preview of the "
        "first few chunks.  This is the endpoint to use for testing the "
        "complete ingestion pipeline before embedding."
    ),
)
async def process_and_chunk_document(
    file: UploadFile = File(..., description="The document file to process."),
    user_id: str = Form(..., description="Owning user's ID."),
    document_id: str = Form(default=None, description="Optional pre-generated document ID."),
    subject: str = Form(default=None, description="Subject label."),
    topic: str = Form(default=None, description="Topic label."),
) -> SuccessResponse:
    """Upload a document and run the full extract → clean → chunk pipeline."""
    contents, _ = await _validate_and_read_file(file)

    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="rag_pipeline_")
        tmp_path = Path(tmp_dir) / file.filename
        tmp_path.write_bytes(contents)

        logger.info(
            "Full pipeline: name=%s  size=%.1fKB  user=%s",
            file.filename,
            len(contents) / 1024,
            user_id,
        )

        chunked = process_document_pipeline(
            file_path=tmp_path,
            user_id=user_id,
            document_id=document_id or None,
            subject=subject or None,
            topic=topic or None,
        )

        # Build chunk previews (first N chunks)
        previews = [
            DocumentChunkPreview(
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                char_count=c.char_count,
                text_preview=c.text[:200] + ("..." if len(c.text) > 200 else ""),
                page_number=c.page_number,
                slide_number=c.slide_number,
                slide_title=c.slide_title,
                heading=c.heading,
            )
            for c in chunked.chunks[:CHUNK_PREVIEW_COUNT]
        ]

        response_data = DocumentProcessAndChunkResponse(
            document_id=chunked.document_id,
            document_name=chunked.document_name,
            file_type=chunked.file_type,
            user_id=chunked.user_id,
            subject=chunked.subject,
            topic=chunked.topic,
            total_pages=len(set(
                c.page_number or c.slide_number or 0 for c in chunked.chunks
            )),
            total_chunks=chunked.total_chunks,
            total_characters=chunked.total_characters,
            chunk_size=chunked.chunk_size,
            chunk_overlap=chunked.chunk_overlap,
            status=chunked.status,
            extraction_time_ms=chunked.extraction_time_ms,
            cleaning_time_ms=chunked.cleaning_time_ms,
            chunking_time_ms=chunked.chunking_time_ms,
            total_processing_time_ms=chunked.total_processing_time_ms,
            chunks_preview=previews,
        )

        return SuccessResponse(
            message=(
                f"Document '{file.filename}' processed into "
                f"{chunked.total_chunks} chunks."
            ),
            data=response_data.model_dump(),
        )

    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Processing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during document processing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during document processing.",
        )
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

