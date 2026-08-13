"""
Plain-text file loader.

Reads .txt files using Python's built-in I/O with encoding detection
fallback.  The entire file is returned as a single ExtractedPage.
"""

from pathlib import Path

from app.core.logging import get_logger
from app.loaders.base_loader import BaseLoader
from app.schemas.document import ExtractedPage

logger = get_logger(__name__)


class TXTLoader(BaseLoader):
    """Extract text from plain .txt files."""

    supported_extensions = {".txt"}

    # Encodings to try in order
    _ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

    def extract(self, file_path: Path) -> list[ExtractedPage]:
        """
        Read a plain-text file and return its content as a single page.

        Parameters
        ----------
        file_path : Path
            Path to the TXT file.

        Returns
        -------
        list[ExtractedPage]
            A single-element list with the full file content.
        """
        self.validate_file(file_path)
        logger.info("TXT extraction started: %s", file_path.name)

        text = self._read_with_fallback(file_path)

        if not text or not text.strip():
            raise ValueError(
                f"No text could be extracted from TXT: {file_path.name}"
            )

        page = ExtractedPage(
            text=text,
            char_count=len(text),
        )

        logger.info(
            "TXT extraction complete: %s — %d characters",
            file_path.name,
            len(text),
        )
        return [page]

    def _read_with_fallback(self, file_path: Path) -> str:
        """Try multiple encodings until one succeeds."""
        for encoding in self._ENCODINGS:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                logger.debug(
                    "Encoding %s failed for %s — trying next",
                    encoding,
                    file_path.name,
                )
                continue

        raise RuntimeError(
            f"Could not decode TXT file '{file_path.name}' with any "
            f"supported encoding: {self._ENCODINGS}"
        )
