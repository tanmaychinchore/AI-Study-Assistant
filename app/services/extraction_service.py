"""
Document extraction orchestrator.

Routes an uploaded file to the correct format-specific loader,
collects the extracted pages, and assembles the canonical
`ExtractedDocument` that downstream stages consume.
"""

import time
import uuid
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger
from app.loaders.base_loader import BaseLoader
from app.loaders.pdf_loader import PDFLoader
from app.loaders.ppt_loader import PPTXLoader
from app.loaders.docx_loader import DOCXLoader
from app.loaders.txt_loader import TXTLoader
from app.schemas.document import (
    ExtractedDocument,
    ExtractedPage,
    FileType,
    ProcessingStatus,
    SUPPORTED_EXTENSIONS,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loader registry — maps FileType to a loader instance
# ---------------------------------------------------------------------------

_LOADERS: dict[FileType, BaseLoader] = {
    FileType.PDF: PDFLoader(),
    FileType.PPTX: PPTXLoader(),
    FileType.DOCX: DOCXLoader(),
    FileType.TXT: TXTLoader(),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_file_type(file_path: Path) -> FileType:
    """
    Determine the FileType from a file's extension.

    Raises
    ------
    ValueError
        If the extension is not in the supported set.
    """
    ext = file_path.suffix.lower()
    file_type = SUPPORTED_EXTENSIONS.get(ext)
    if file_type is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported formats: {supported}"
        )
    return file_type


def extract_document(
    file_path: Path,
    user_id: str,
    document_id: Optional[str] = None,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
) -> ExtractedDocument:
    """
    Run the full extraction pipeline for a single document.

    1. Validate the file extension.
    2. Select the correct loader.
    3. Extract pages/slides/sections.
    4. Assemble an `ExtractedDocument`.

    Parameters
    ----------
    file_path : Path
        Absolute path to the uploaded file.
    user_id : str
        Owning user's ID (required for data isolation).
    document_id : str, optional
        Pre-generated document ID.  A UUID is created if omitted.
    subject : str, optional
        Subject label.
    topic : str, optional
        Topic label.

    Returns
    -------
    ExtractedDocument
        Complete extraction result ready for cleaning and chunking.

    Raises
    ------
    ValueError
        Unsupported format, empty file, or no extractable text.
    FileNotFoundError
        File does not exist.
    RuntimeError
        Extraction failed (corrupt file, library error, etc.).
    """
    start_time = time.perf_counter()

    # --- Resolve file type ---
    file_type = get_file_type(file_path)
    logger.info(
        "Extraction requested: file=%s  type=%s  user=%s",
        file_path.name,
        file_type.value,
        user_id,
    )

    # --- Select loader ---
    loader = _LOADERS.get(file_type)
    if loader is None:
        raise ValueError(f"No loader registered for file type: {file_type.value}")

    # --- Extract ---
    pages: list[ExtractedPage] = loader.extract(file_path)

    # --- Assemble document ---
    doc_id = document_id or str(uuid.uuid4())
    total_chars = sum(p.char_count for p in pages)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    extracted = ExtractedDocument(
        document_id=doc_id,
        document_name=file_path.name,
        file_type=file_type,
        user_id=user_id,
        subject=subject,
        topic=topic,
        pages=pages,
        total_pages=len(pages),
        total_characters=total_chars,
        status=ProcessingStatus.EXTRACTED,
        extraction_time_ms=round(elapsed_ms, 2),
    )

    logger.info(
        "Extraction complete: doc_id=%s  pages=%d  chars=%d  time=%.2fms",
        doc_id,
        len(pages),
        total_chars,
        elapsed_ms,
    )

    return extracted
