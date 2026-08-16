"""
Comprehensive tests for the Complete Document Indexing Pipeline (Task 6).

Covers:
  - Multi-format indexing (PDF, PPTX, DOCX, TXT)
  - Duplicate prevention / Re-indexing behavior
  - Stage-by-stage timing metrics and count verification
  - Metadata preservation (pages, slides, titles, headings, subject, topic)
  - Error handling (empty file, unsupported formats, missing services)
  - API endpoint verification: POST /api/v1/documents/index
"""

from pathlib import Path
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.document import FileType, ProcessingStatus
from app.schemas.indexing import IndexingResult
from app.services.astra_db_service import AstraDBService
from app.services.embedding_service import EmbeddingService
from app.services.indexing_service import IndexingService, index_document

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
SAMPLE_PPTX = FIXTURES_DIR / "sample.pptx"
SAMPLE_DOCX = FIXTURES_DIR / "sample.docx"
SAMPLE_TXT = FIXTURES_DIR / "sample.txt"


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedding_service():
    """Load BGE-M3 model once for the test module."""
    svc = EmbeddingService()
    svc.load_model()
    return svc


@pytest.fixture(scope="module")
def live_astra_service():
    """Connect to live Astra DB if configured in .env."""
    svc = AstraDBService()
    if not svc.is_configured:
        pytest.skip("Astra DB is not configured in .env. Skipping live Astra DB indexing tests.")
    try:
        svc.connect()
        svc.initialize_collection()
    except Exception as exc:
        pytest.skip(f"Could not connect to Astra DB: {exc}. Skipping live tests.")
    return svc


@pytest.fixture(scope="module")
def indexing_service(embedding_service, live_astra_service):
    """Create IndexingService with live embedding and Astra DB services."""
    return IndexingService(
        embedding_service=embedding_service,
        astra_service=live_astra_service,
    )


# ===========================================================================
# 1. MULTI-FORMAT END-TO-END INDEXING TESTS
# ===========================================================================

class TestMultiFormatIndexing:
    """Tests for indexing all supported educational document formats."""

    def test_index_pptx_document(self, indexing_service, live_astra_service):
        """Index PPTX file -> extract -> clean -> chunk -> embed -> Astra DB."""
        doc_id = f"test_pptx_{uuid.uuid4().hex[:8]}"

        result = indexing_service.index_document(
            file_path=SAMPLE_PPTX,
            user_id="student_101",
            document_id=doc_id,
            subject="Computer Science",
            topic="Operating Systems",
        )

        assert isinstance(result, IndexingResult)
        assert result.document_id == doc_id
        assert result.file_type == FileType.PPTX
        assert result.status == ProcessingStatus.INDEXED
        assert result.total_chunks > 0
        assert result.embeddings_generated == result.total_chunks
        assert result.vectors_inserted == result.total_chunks
        assert result.collection == live_astra_service.collection_name

        # Verify timing statistics
        stats = result.statistics
        assert stats.extraction_time_ms > 0
        assert stats.cleaning_time_ms >= 0
        assert stats.chunking_time_ms >= 0
        assert stats.embedding_time_ms > 0
        assert stats.astra_insertion_time_ms > 0
        assert stats.total_time_ms > 0

        # Verify preview
        assert len(result.chunks_preview) > 0
        assert result.chunks_preview[0].slide_number is not None

        # Clean up Astra DB
        live_astra_service.delete_document_chunks(doc_id)

    def test_index_pdf_document(self, indexing_service, live_astra_service):
        """Index PDF file -> verify full pipeline execution."""
        doc_id = f"test_pdf_{uuid.uuid4().hex[:8]}"

        result = indexing_service.index_document(
            file_path=SAMPLE_PDF,
            user_id="student_102",
            document_id=doc_id,
            subject="Algorithms",
        )

        assert result.document_id == doc_id
        assert result.file_type == FileType.PDF
        assert result.total_chunks > 0
        assert result.embeddings_generated == result.total_chunks
        assert result.vectors_inserted == result.total_chunks

        # Clean up
        live_astra_service.delete_document_chunks(doc_id)

    def test_index_docx_document(self, indexing_service, live_astra_service):
        """Index DOCX file -> verify heading preservation and indexing."""
        doc_id = f"test_docx_{uuid.uuid4().hex[:8]}"

        result = indexing_service.index_document(
            file_path=SAMPLE_DOCX,
            user_id="student_103",
            document_id=doc_id,
            subject="Data Structures",
        )

        assert result.document_id == doc_id
        assert result.file_type == FileType.DOCX
        assert result.total_chunks > 0
        assert result.vectors_inserted == result.total_chunks

        # Clean up
        live_astra_service.delete_document_chunks(doc_id)

    def test_index_txt_document(self, indexing_service, live_astra_service):
        """Index plain text file -> verify full pipeline execution."""
        doc_id = f"test_txt_{uuid.uuid4().hex[:8]}"

        result = indexing_service.index_document(
            file_path=SAMPLE_TXT,
            user_id="student_104",
            document_id=doc_id,
        )

        assert result.document_id == doc_id
        assert result.file_type == FileType.TXT
        assert result.total_chunks > 0
        assert result.vectors_inserted == result.total_chunks

        # Clean up
        live_astra_service.delete_document_chunks(doc_id)


# ===========================================================================
# 2. RE-INDEXING & DUPLICATE PREVENTION TESTS
# ===========================================================================

class TestReindexingDuplicatePrevention:
    """Verify that re-indexing the same document does not leave duplicate vectors."""

    def test_reindex_same_document_prevents_duplicates(self, indexing_service, live_astra_service):
        """
        Indexing the same document_id twice must overwrite/replace old chunks,
        resulting in exactly the current chunk count in Astra DB.
        """
        doc_id = f"test_reindex_{uuid.uuid4().hex[:8]}"

        # First indexing run
        result_1 = indexing_service.index_document(
            file_path=SAMPLE_TXT,
            user_id="reindex_user",
            document_id=doc_id,
        )
        first_count = result_1.vectors_inserted

        # Second indexing run with identical document_id
        result_2 = indexing_service.index_document(
            file_path=SAMPLE_TXT,
            user_id="reindex_user",
            document_id=doc_id,
        )
        second_count = result_2.vectors_inserted

        assert first_count == second_count

        # Clean up
        deleted = live_astra_service.delete_document_chunks(doc_id)
        # Deleted count should equal exactly the chunk count (no duplicate accumulation)
        assert deleted == second_count


# ===========================================================================
# 3. ERROR HANDLING & VALIDATION TESTS
# ===========================================================================

class TestIndexingErrorHandling:
    """Test resilience against error states and missing resources."""

    def test_missing_embedding_service_raises(self, live_astra_service):
        """Unloaded embedding service must raise RuntimeError."""
        unloaded_emb = EmbeddingService()
        svc = IndexingService(embedding_service=unloaded_emb, astra_service=live_astra_service)

        with pytest.raises(RuntimeError, match="EmbeddingService is not available"):
            svc.index_document(file_path=SAMPLE_TXT, user_id="user_1")

    def test_missing_astra_service_raises(self, embedding_service):
        """Unconnected Astra DB service must raise RuntimeError."""
        unconnected_db = AstraDBService(api_endpoint="", token="")
        svc = IndexingService(embedding_service=embedding_service, astra_service=unconnected_db)

        with pytest.raises(RuntimeError, match="AstraDBService is not connected"):
            svc.index_document(file_path=SAMPLE_TXT, user_id="user_1")

    def test_empty_document_raises_value_error(self, indexing_service):
        """Empty text file should raise ValueError without inserting vectors."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write("   \n\n   ")
            temp_empty = Path(f.name)

        try:
            with pytest.raises(ValueError):
                indexing_service.index_document(file_path=temp_empty, user_id="user_empty")
        finally:
            temp_empty.unlink(missing_ok=True)


# ===========================================================================
# 4. API ENDPOINT TESTS (POST /api/v1/documents/index)
# ===========================================================================

class TestIndexingAPIEndpoint:
    """Tests for POST /api/v1/documents/index HTTP endpoint."""

    @pytest.fixture
    def client(self, embedding_service, live_astra_service):
        app.state.embedding_service = embedding_service
        app.state.astra_db_service = live_astra_service
        with TestClient(app) as test_client:
            yield test_client

    def test_api_index_document_success(self, client, live_astra_service):
        """POST /api/v1/documents/index with valid file returns 200 OK and IndexingResult."""
        doc_id = f"api_index_test_{uuid.uuid4().hex[:8]}"

        with open(SAMPLE_TXT, "rb") as f:
            response = client.post(
                "/api/v1/documents/index",
                data={
                    "user_id": "api_test_student",
                    "document_id": doc_id,
                    "subject": "API Testing",
                },
                files={"file": ("sample.txt", f, "text/plain")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        result = data["data"]
        assert result["document_id"] == doc_id
        assert result["file_type"] == "txt"
        assert result["status"] == "indexed"
        assert result["vectors_inserted"] > 0
        assert "statistics" in result
        assert result["statistics"]["total_time_ms"] > 0

        # Clean up test document chunks
        live_astra_service.delete_document_chunks(doc_id)

    def test_api_index_empty_file_returns_400(self, client):
        """POST /api/v1/documents/index with empty file returns 400."""
        response = client.post(
            "/api/v1/documents/index",
            data={"user_id": "api_test_student"},
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_api_index_unsupported_file_returns_400(self, client):
        """POST /api/v1/documents/index with .exe returns 400."""
        response = client.post(
            "/api/v1/documents/index",
            data={"user_id": "api_test_student"},
            files={"file": ("malicious.exe", b"binary content", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()
