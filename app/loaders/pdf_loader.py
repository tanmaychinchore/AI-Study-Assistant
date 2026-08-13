"""
PDF document loader using PyMuPDF (fitz).

Extracts text page-by-page, preserving page numbers and character
counts for downstream metadata propagation.
"""

from pathlib import Path

import fitz  # PyMuPDF

from app.core.logging import get_logger
from app.loaders.base_loader import BaseLoader
from app.schemas.document import ExtractedPage

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    """Extract text from PDF files using PyMuPDF."""

    supported_extensions = {".pdf"}

    def extract(self, file_path: Path) -> list[ExtractedPage]:
        """
        Extract text from every page of a PDF.

        Parameters
        ----------
        file_path : Path
            Path to the PDF file.

        Returns
        -------
        list[ExtractedPage]
            One ExtractedPage per PDF page with page_number (1-indexed).
        """
        self.validate_file(file_path)
        logger.info("PDF extraction started: %s", file_path.name)

        pages: list[ExtractedPage] = []

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open PDF '{file_path.name}': {exc}"
            ) from exc

        try:
            if doc.page_count == 0:
                raise ValueError(f"PDF has no pages: {file_path.name}")

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text("text")  # plain-text extraction

                # Skip completely blank pages but keep page numbering intact
                if not text or not text.strip():
                    logger.debug(
                        "Page %d is blank in %s — skipping",
                        page_num + 1,
                        file_path.name,
                    )
                    continue

                pages.append(
                    ExtractedPage(
                        page_number=page_num + 1,  # 1-indexed
                        text=text,
                        char_count=len(text),
                    )
                )
        finally:
            doc.close()

        if not pages:
            raise ValueError(
                f"No text could be extracted from PDF: {file_path.name}"
            )

        logger.info(
            "PDF extraction complete: %s — %d pages extracted",
            file_path.name,
            len(pages),
        )
        return pages
