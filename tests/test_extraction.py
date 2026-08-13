"""
Comprehensive tests for the document extraction pipeline (Task 2).

Covers:
  - PDF extraction (PyMuPDF)
  - PPTX extraction (python-pptx)
  - DOCX extraction (python-docx)
  - TXT extraction (native Python)
  - File validation (unsupported, empty, missing)
  - Extraction service orchestrator
  - Metadata preservation
  - Error handling
"""

import tempfile
from pathlib import Path

import pytest

from app.loaders.pdf_loader import PDFLoader
from app.loaders.ppt_loader import PPTXLoader
from app.loaders.docx_loader import DOCXLoader
from app.loaders.txt_loader import TXTLoader
from app.schemas.document import ExtractedPage, FileType, ProcessingStatus
from app.services.extraction_service import extract_document, get_file_type


# ---------------------------------------------------------------------------
# Paths to fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
SAMPLE_PPTX = FIXTURES_DIR / "sample.pptx"
SAMPLE_DOCX = FIXTURES_DIR / "sample.docx"
SAMPLE_TXT = FIXTURES_DIR / "sample.txt"
EMPTY_TXT = FIXTURES_DIR / "empty.txt"


# ===========================================================================
# 1. PDF EXTRACTION
# ===========================================================================

class TestPDFLoader:
    """Tests for PDF extraction using PyMuPDF."""

    def setup_method(self):
        self.loader = PDFLoader()

    def test_extract_valid_pdf(self):
        """A valid multi-page PDF should yield one ExtractedPage per page."""
        pages = self.loader.extract(SAMPLE_PDF)

        assert len(pages) == 3, f"Expected 3 pages, got {len(pages)}"
        for page in pages:
            assert isinstance(page, ExtractedPage)
            assert page.page_number is not None
            assert page.page_number >= 1
            assert len(page.text) > 0
            assert page.char_count == len(page.text)

    def test_pdf_page_numbers_are_sequential(self):
        """Page numbers should be 1-indexed and sequential."""
        pages = self.loader.extract(SAMPLE_PDF)
        page_numbers = [p.page_number for p in pages]
        assert page_numbers == [1, 2, 3]

    def test_pdf_contains_expected_content(self):
        """Extracted text should contain known keywords from the fixture."""
        pages = self.loader.extract(SAMPLE_PDF)
        all_text = " ".join(p.text for p in pages)

        assert "Operating System" in all_text or "operating system" in all_text.lower()
        assert "process" in all_text.lower()
        assert "deadlock" in all_text.lower()

    def test_pdf_metadata_fields(self):
        """PDF pages should only have page_number set, not slide fields."""
        pages = self.loader.extract(SAMPLE_PDF)
        for page in pages:
            assert page.page_number is not None
            assert page.slide_number is None
            assert page.slide_title is None

    def test_pdf_file_not_found(self):
        """Missing PDF should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.loader.extract(Path("nonexistent.pdf"))

    def test_pdf_unsupported_extension(self):
        """Wrong extension should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"not a pdf")
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Unsupported extension"):
                self.loader.extract(tmp)
        finally:
            tmp.unlink(missing_ok=True)


# ===========================================================================
# 2. PPTX EXTRACTION
# ===========================================================================

class TestPPTXLoader:
    """Tests for PPTX extraction using python-pptx."""

    def setup_method(self):
        self.loader = PPTXLoader()

    def test_extract_valid_pptx(self):
        """A valid PPTX should yield one ExtractedPage per slide."""
        pages = self.loader.extract(SAMPLE_PPTX)

        assert len(pages) == 3, f"Expected 3 slides, got {len(pages)}"
        for page in pages:
            assert isinstance(page, ExtractedPage)
            assert page.slide_number is not None
            assert page.slide_number >= 1
            assert len(page.text) > 0
            assert page.char_count == len(page.text)

    def test_pptx_slide_numbers_are_sequential(self):
        """Slide numbers should be 1-indexed and sequential."""
        pages = self.loader.extract(SAMPLE_PPTX)
        slide_numbers = [p.slide_number for p in pages]
        assert slide_numbers == [1, 2, 3]

    def test_pptx_captures_slide_titles(self):
        """Slide titles should be extracted where available."""
        pages = self.loader.extract(SAMPLE_PPTX)
        titles = [p.slide_title for p in pages]

        # Our fixture has titles on all slides
        assert any(t is not None for t in titles), "At least one slide should have a title"
        assert "Database Fundamentals" in titles
        assert "SQL Basics" in titles

    def test_pptx_contains_expected_content(self):
        """Extracted text should contain known keywords."""
        pages = self.loader.extract(SAMPLE_PPTX)
        all_text = " ".join(p.text for p in pages)

        assert "database" in all_text.lower()
        assert "sql" in all_text.lower()
        assert "normalization" in all_text.lower()

    def test_pptx_metadata_fields(self):
        """PPTX pages should have slide_number set, not page_number."""
        pages = self.loader.extract(SAMPLE_PPTX)
        for page in pages:
            assert page.slide_number is not None
            assert page.page_number is None

    def test_legacy_ppt_rejected(self):
        """A .ppt file should be rejected with a clear message."""
        with tempfile.NamedTemporaryFile(suffix=".ppt", delete=False) as f:
            f.write(b"fake ppt content")
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Legacy .ppt"):
                self.loader.extract(tmp)
        finally:
            tmp.unlink(missing_ok=True)


# ===========================================================================
# 3. DOCX EXTRACTION
# ===========================================================================

class TestDOCXLoader:
    """Tests for DOCX extraction using python-docx."""

    def setup_method(self):
        self.loader = DOCXLoader()

    def test_extract_valid_docx(self):
        """A valid DOCX with headings should yield multiple sections."""
        pages = self.loader.extract(SAMPLE_DOCX)

        assert len(pages) >= 1, "Should extract at least one section"
        for page in pages:
            assert isinstance(page, ExtractedPage)
            assert len(page.text) > 0
            assert page.char_count == len(page.text)

    def test_docx_preserves_headings(self):
        """Sections should capture heading text."""
        pages = self.loader.extract(SAMPLE_DOCX)
        headings = [p.heading for p in pages if p.heading]

        assert len(headings) > 0, "Should detect at least one heading"
        heading_text = " ".join(headings).lower()
        assert "arrays" in heading_text or "linked" in heading_text or "data structures" in heading_text

    def test_docx_contains_expected_content(self):
        """Extracted text should contain known keywords."""
        pages = self.loader.extract(SAMPLE_DOCX)
        all_text = " ".join(p.text for p in pages)

        assert "data structure" in all_text.lower()
        assert "array" in all_text.lower()

    def test_docx_metadata_fields(self):
        """DOCX pages should have heading set, not page/slide numbers."""
        pages = self.loader.extract(SAMPLE_DOCX)
        for page in pages:
            assert page.page_number is None
            assert page.slide_number is None

    def test_legacy_doc_rejected(self):
        """A .doc file should be rejected with a clear message."""
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as f:
            f.write(b"fake doc content")
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Legacy .doc"):
                self.loader.extract(tmp)
        finally:
            tmp.unlink(missing_ok=True)


# ===========================================================================
# 4. TXT EXTRACTION
# ===========================================================================

class TestTXTLoader:
    """Tests for TXT extraction using native Python."""

    def setup_method(self):
        self.loader = TXTLoader()

    def test_extract_valid_txt(self):
        """A valid TXT file should yield exactly one ExtractedPage."""
        pages = self.loader.extract(SAMPLE_TXT)

        assert len(pages) == 1, "TXT should produce exactly one page"
        page = pages[0]
        assert isinstance(page, ExtractedPage)
        assert len(page.text) > 0
        assert page.char_count == len(page.text)

    def test_txt_contains_expected_content(self):
        """Extracted text should contain the fixture content."""
        pages = self.loader.extract(SAMPLE_TXT)
        text = pages[0].text

        assert "process scheduling" in text.lower()
        assert "deadlock" in text.lower()
        assert "virtual memory" in text.lower()

    def test_txt_metadata_fields(self):
        """TXT page should have no page/slide/heading metadata."""
        pages = self.loader.extract(SAMPLE_TXT)
        page = pages[0]
        assert page.page_number is None
        assert page.slide_number is None
        assert page.slide_title is None

    def test_txt_empty_file_raises(self):
        """An empty/whitespace-only TXT file should raise ValueError."""
        with pytest.raises(ValueError, match="No text"):
            self.loader.extract(EMPTY_TXT)

    def test_txt_file_not_found(self):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.loader.extract(Path("nonexistent.txt"))


# ===========================================================================
# 5. FILE TYPE DETECTION
# ===========================================================================

class TestFileTypeDetection:
    """Tests for the get_file_type utility."""

    def test_pdf_detection(self):
        assert get_file_type(Path("doc.pdf")) == FileType.PDF

    def test_pptx_detection(self):
        assert get_file_type(Path("slides.pptx")) == FileType.PPTX

    def test_ppt_detection(self):
        assert get_file_type(Path("old.ppt")) == FileType.PPTX

    def test_docx_detection(self):
        assert get_file_type(Path("report.docx")) == FileType.DOCX

    def test_doc_detection(self):
        assert get_file_type(Path("old.doc")) == FileType.DOCX

    def test_txt_detection(self):
        assert get_file_type(Path("notes.txt")) == FileType.TXT

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            get_file_type(Path("image.png"))

    def test_case_insensitive(self):
        assert get_file_type(Path("DOC.PDF")) == FileType.PDF
        assert get_file_type(Path("SLIDES.PPTX")) == FileType.PPTX


# ===========================================================================
# 6. EXTRACTION SERVICE (ORCHESTRATOR)
# ===========================================================================

class TestExtractionService:
    """Tests for the top-level extract_document orchestrator."""

    def test_extract_pdf_via_service(self):
        """Service should correctly route PDF and return ExtractedDocument."""
        doc = extract_document(
            file_path=SAMPLE_PDF,
            user_id="test_user_001",
            subject="Operating Systems",
            topic="Introduction",
        )

        assert doc.file_type == FileType.PDF
        assert doc.user_id == "test_user_001"
        assert doc.subject == "Operating Systems"
        assert doc.topic == "Introduction"
        assert doc.document_name == "sample.pdf"
        assert doc.total_pages == 3
        assert doc.total_characters > 0
        assert doc.status == ProcessingStatus.EXTRACTED
        assert doc.extraction_time_ms is not None
        assert doc.extraction_time_ms > 0
        assert len(doc.pages) == 3

    def test_extract_pptx_via_service(self):
        """Service should correctly route PPTX."""
        doc = extract_document(
            file_path=SAMPLE_PPTX,
            user_id="test_user_002",
        )

        assert doc.file_type == FileType.PPTX
        assert doc.total_pages == 3
        assert len(doc.pages) == 3

    def test_extract_docx_via_service(self):
        """Service should correctly route DOCX."""
        doc = extract_document(
            file_path=SAMPLE_DOCX,
            user_id="test_user_003",
        )

        assert doc.file_type == FileType.DOCX
        assert doc.total_pages >= 1
        assert doc.total_characters > 0

    def test_extract_txt_via_service(self):
        """Service should correctly route TXT."""
        doc = extract_document(
            file_path=SAMPLE_TXT,
            user_id="test_user_004",
        )

        assert doc.file_type == FileType.TXT
        assert doc.total_pages == 1
        assert doc.total_characters > 0

    def test_auto_generated_document_id(self):
        """When no document_id is provided, a UUID should be generated."""
        doc = extract_document(
            file_path=SAMPLE_TXT,
            user_id="test_user_005",
        )

        assert doc.document_id is not None
        assert len(doc.document_id) > 0

    def test_custom_document_id(self):
        """A provided document_id should be used as-is."""
        doc = extract_document(
            file_path=SAMPLE_TXT,
            user_id="test_user_006",
            document_id="my-custom-id-123",
        )

        assert doc.document_id == "my-custom-id-123"

    def test_user_id_preserved(self):
        """user_id should be preserved exactly for data isolation."""
        doc = extract_document(
            file_path=SAMPLE_TXT,
            user_id="user_abc_xyz",
        )

        assert doc.user_id == "user_abc_xyz"

    def test_total_characters_is_sum(self):
        """total_characters should equal the sum of all page char_counts."""
        doc = extract_document(
            file_path=SAMPLE_PDF,
            user_id="test_user_007",
        )

        expected = sum(p.char_count for p in doc.pages)
        assert doc.total_characters == expected

    def test_unsupported_file_raises(self):
        """Unsupported extension should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            f.write(b"fake content")
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Unsupported file extension"):
                extract_document(file_path=tmp, user_id="test_user")
        finally:
            tmp.unlink(missing_ok=True)

    def test_missing_file_raises(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_document(
                file_path=Path("does_not_exist.pdf"),
                user_id="test_user",
            )
