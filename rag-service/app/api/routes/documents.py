"""
Document processing API routes.

Provides the endpoint for uploading and extracting documents.
The actual extraction is delegated to the extraction service.
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
from app.schemas.response import ErrorResponse, SuccessResponse
from app.services.extraction_service import extract_document, get_file_type

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# Maximum file size: 50 MB
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


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
    """
    Upload a document, extract its text and metadata.

    The file is temporarily saved to disk, processed by the appropriate
    format loader, and then cleaned up.  The extraction result is returned
    as structured JSON.
    """
    # --- Validate filename & extension ---
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

    # --- Read file and check size ---
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

    # --- Save to temp file and extract ---
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

        # Build response
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except FileNotFoundError as exc:
        logger.error("File not found during extraction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
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
        # Clean up temp files
        if tmp_dir:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                logger.warning("Failed to clean up temp directory: %s", tmp_dir)
