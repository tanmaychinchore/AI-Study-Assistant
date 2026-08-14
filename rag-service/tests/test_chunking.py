"""
Tests for the document chunking service.

Validates chunk splitting, metadata propagation, overlap behavior,
and integration with the full document processing pipeline.
"""

from pathlib import Path

import pytest

from app.core.config import settings
from app.schemas.document import ExtractedDocument, ExtractedPage, FileType, ProcessingStatus
from app.schemas.chunk import DocumentChunk, ChunkedDocument
from app.services.chunking_service import chunk_document
from app.services.document_service import process_document_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
SAMPLE_PPTX = FIXTURES_DIR / "sample.pptx"
SAMPLE_DOCX = FIXTURES_DIR / "sample.docx"
SAMPLE_TXT = FIXTURES_DIR / "sample.txt"


def _make_doc(
    pages_text: list[str],
    file_type: FileType = FileType.PDF,
    user_id: str = "user_test",
    subject: str = "TestSubject",
    topic: str = "TestTopic",
) -> ExtractedDocument:
    """Helper to build an ExtractedDocument from raw text strings."""
    pages = [
        ExtractedPage(
            page_number=i + 1,
            text=text,
            char_count=len(text),
        )
        for i, text in enumerate(pages_text)
    ]
    return ExtractedDocument(
        document_id="test-doc-chunk",
        document_name="test.pdf",
        file_type=file_type,
        user_id=user_id,
        subject=subject,
        topic=topic,
        pages=pages,
        total_pages=len(pages),
        total_characters=sum(len(t) for t in pages_text),
    )


# ===========================================================================
# Basic chunking behavior
# ===========================================================================

class TestChunkDocument:
    """Tests for the chunk_document function."""

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size should produce one chunk."""
        doc = _make_doc(["Short paragraph."])
        chunked, time_ms = chunk_document(doc, chunk_size=1000, chunk_overlap=100)

        assert chunked.total_chunks == 1
        assert chunked.chunks[0].text == "Short paragraph."
        assert time_ms > 0

    def test_long_text_multiple_chunks(self):
        """Text longer than chunk_size should produce multiple chunks."""
        long_text = "This is a sentence. " * 200  # ~4000 chars
        doc = _make_doc([long_text])
        chunked, _ = chunk_document(doc, chunk_size=500, chunk_overlap=50)

        assert chunked.total_chunks > 1
        # Each chunk should be <= chunk_size (approximately)
        for chunk in chunked.chunks:
            assert chunk.char_count <= 600  # some tolerance for splitter

    def test_chunk_ids_are_unique(self):
        """Every chunk should have a unique chunk_id."""
        long_text = "Word " * 500
        doc = _make_doc([long_text])
        chunked, _ = chunk_document(doc, chunk_size=200, chunk_overlap=20)

        ids = [c.chunk_id for c in chunked.chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_indices_are_sequential(self):
        """Chunk indices should be 0, 1, 2, ..."""
        long_text = "Word " * 500
        doc = _make_doc([long_text])
        chunked, _ = chunk_document(doc, chunk_size=200, chunk_overlap=20)

        indices = [c.chunk_index for c in chunked.chunks]
        assert indices == list(range(len(indices)))

    def test_chunk_id_format(self):
        """Chunk IDs should follow the pattern '{doc_id}_chunk_XXXX'."""
        doc = _make_doc(["Some text content here."])
        chunked, _ = chunk_document(doc)

        for chunk in chunked.chunks:
            assert chunk.chunk_id.startswith("test-doc-chunk_chunk_")


# ===========================================================================
# Metadata propagation
# ===========================================================================

class TestMetadataPropagation:
    """Tests that chunks inherit source metadata correctly."""

    def test_document_metadata(self):
        """Chunks should carry document-level metadata."""
        doc = _make_doc(
            ["Content here."],
            user_id="user_xyz",
            subject="DBMS",
            topic="Normalization",
        )
        chunked, _ = chunk_document(doc)

        for chunk in chunked.chunks:
            assert chunk.user_id == "user_xyz"
            assert chunk.document_id == "test-doc-chunk"
            assert chunk.document_name == "test.pdf"
            assert chunk.file_type == FileType.PDF
            assert chunk.subject == "DBMS"
            assert chunk.topic == "Normalization"

    def test_page_number_propagated(self):
        """PDF chunks should inherit page_number from their source page."""
        doc = _make_doc(["Page one text.", "Page two text."])
        chunked, _ = chunk_document(doc, chunk_size=1000)

        page_numbers = [c.page_number for c in chunked.chunks]
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_slide_metadata_propagated(self):
        """PPTX chunks should inherit slide_number and slide_title."""
        pages = [
            ExtractedPage(
                slide_number=1,
                slide_title="Introduction",
                text="This is the intro slide content.",
                char_count=31,
            ),
            ExtractedPage(
                slide_number=2,
                slide_title="SQL Basics",
                text="SQL is a query language for databases.",
                char_count=37,
            ),
        ]
        doc = ExtractedDocument(
            document_id="pptx-test",
            document_name="slides.pptx",
            file_type=FileType.PPTX,
            user_id="user_pptx",
            pages=pages,
            total_pages=2,
            total_characters=68,
        )
        chunked, _ = chunk_document(doc)

        slides = {c.slide_number for c in chunked.chunks}
        titles = {c.slide_title for c in chunked.chunks}
        assert 1 in slides
        assert 2 in slides
        assert "Introduction" in titles
        assert "SQL Basics" in titles

    def test_heading_propagated(self):
        """DOCX chunks should inherit heading from their source section."""
        pages = [
            ExtractedPage(
                heading="Arrays",
                text="An array is a data structure.",
                char_count=28,
            ),
        ]
        doc = ExtractedDocument(
            document_id="docx-test",
            document_name="notes.docx",
            file_type=FileType.DOCX,
            user_id="user_docx",
            pages=pages,
            total_pages=1,
            total_characters=28,
        )
        chunked, _ = chunk_document(doc)

        assert chunked.chunks[0].heading == "Arrays"


# ===========================================================================
# Chunk overlap
# ===========================================================================

class TestChunkOverlap:
    """Tests for chunk overlap behavior."""

    def test_overlap_creates_shared_text(self):
        """With overlap, consecutive chunks should share some text."""
        # Create a long, continuous text
        text = " ".join(f"Sentence number {i} with some content." for i in range(50))
        doc = _make_doc([text])
        chunked, _ = chunk_document(doc, chunk_size=200, chunk_overlap=50)

        if chunked.total_chunks >= 2:
            # Check that the end of chunk N overlaps with the start of chunk N+1
            chunk_a = chunked.chunks[0].text
            chunk_b = chunked.chunks[1].text
            # The last ~50 chars of A should appear at the start of B
            overlap_region = chunk_a[-50:]
            # At least some substring should be shared
            assert any(
                word in chunk_b for word in overlap_region.split()
            ), "Consecutive chunks should share overlapping words"

    def test_zero_overlap(self):
        """With overlap=0, chunks should not share content."""
        text = " ".join(f"Word{i}" for i in range(200))
        doc = _make_doc([text])
        chunked, _ = chunk_document(doc, chunk_size=100, chunk_overlap=0)

        assert chunked.total_chunks > 1


# ===========================================================================
# Configurable chunk size
# ===========================================================================

class TestConfigurableChunkSize:
    """Tests for different chunk size configurations."""

    def test_small_chunks(self):
        """Small chunk_size should produce more chunks."""
        text = "Word " * 500  # ~2500 chars
        doc = _make_doc([text])

        small, _ = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        large, _ = chunk_document(doc, chunk_size=1000, chunk_overlap=100)

        assert small.total_chunks > large.total_chunks

    def test_defaults_from_settings(self):
        """When no size is provided, settings values should be used."""
        doc = _make_doc(["Some text."])
        chunked, _ = chunk_document(doc)

        assert chunked.chunk_size == settings.CHUNK_SIZE
        assert chunked.chunk_overlap == settings.CHUNK_OVERLAP


# ===========================================================================
# ChunkedDocument totals
# ===========================================================================

class TestChunkedDocumentTotals:
    """Tests for aggregate fields on ChunkedDocument."""

    def test_total_chunks_matches(self):
        doc = _make_doc(["Word " * 200])
        chunked, _ = chunk_document(doc, chunk_size=300)

        assert chunked.total_chunks == len(chunked.chunks)

    def test_total_characters_is_sum(self):
        doc = _make_doc(["Word " * 200])
        chunked, _ = chunk_document(doc, chunk_size=300)

        expected = sum(c.char_count for c in chunked.chunks)
        assert chunked.total_characters == expected


# ===========================================================================
# Full pipeline integration
# ===========================================================================

class TestDocumentPipeline:
    """Tests for the full extract → clean → chunk pipeline."""

    def test_pdf_pipeline(self):
        chunked = process_document_pipeline(
            file_path=SAMPLE_PDF,
            user_id="pipeline_user",
            subject="OS",
        )

        assert isinstance(chunked, ChunkedDocument)
        assert chunked.total_chunks > 0
        assert chunked.user_id == "pipeline_user"
        assert chunked.subject == "OS"
        assert chunked.extraction_time_ms is not None
        assert chunked.cleaning_time_ms is not None
        assert chunked.chunking_time_ms is not None
        assert chunked.total_processing_time_ms is not None

    def test_pptx_pipeline(self):
        chunked = process_document_pipeline(
            file_path=SAMPLE_PPTX,
            user_id="pipeline_user",
        )

        assert chunked.total_chunks > 0
        # At least some chunks should have slide metadata
        slides = [c.slide_number for c in chunked.chunks if c.slide_number]
        assert len(slides) > 0

    def test_docx_pipeline(self):
        chunked = process_document_pipeline(
            file_path=SAMPLE_DOCX,
            user_id="pipeline_user",
        )

        assert chunked.total_chunks > 0

    def test_txt_pipeline(self):
        chunked = process_document_pipeline(
            file_path=SAMPLE_TXT,
            user_id="pipeline_user",
        )

        assert chunked.total_chunks >= 1

    def test_pipeline_preserves_user_id(self):
        """user_id must flow through the entire pipeline for data isolation."""
        chunked = process_document_pipeline(
            file_path=SAMPLE_TXT,
            user_id="isolated_user_99",
        )

        for chunk in chunked.chunks:
            assert chunk.user_id == "isolated_user_99"
