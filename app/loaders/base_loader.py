"""
Abstract base loader for document extraction.

All format-specific loaders inherit from BaseLoader and implement
the `extract` method.  This guarantees a uniform interface for the
extraction service regardless of file type.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.document import ExtractedPage


class BaseLoader(ABC):
    """
    Contract every document loader must satisfy.

    Subclasses implement `extract()` which reads a file and returns a
    list of `ExtractedPage` objects.  The orchestrating extraction
    service wraps these pages into a full `ExtractedDocument`.
    """

    # File extensions this loader handles (e.g. {".pdf"})
    supported_extensions: set[str] = set()

    @abstractmethod
    def extract(self, file_path: Path) -> list[ExtractedPage]:
        """
        Extract text and metadata from the file at *file_path*.

        Parameters
        ----------
        file_path : Path
            Absolute path to the uploaded file on disk.

        Returns
        -------
        list[ExtractedPage]
            Ordered list of pages/slides/sections with text and metadata.

        Raises
        ------
        ValueError
            If the file is empty or cannot be read.
        RuntimeError
            If extraction fails for any reason.
        """
        ...

    def validate_file(self, file_path: Path) -> None:
        """
        Basic pre-extraction validation.

        Checks that the file exists, is not empty, and has a supported
        extension.  Individual loaders may override this for additional
        format-specific validation.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"File is empty: {file_path.name}")

        ext = file_path.suffix.lower()
        if self.supported_extensions and ext not in self.supported_extensions:
            raise ValueError(
                f"Unsupported extension '{ext}' for {self.__class__.__name__}. "
                f"Expected one of: {self.supported_extensions}"
            )
