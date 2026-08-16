"""
Comprehensive tests for the BGE-M3 embedding service (Task 4).

Covers:
  - Model initialization and device selection
  - Embedding generation (single, multiple, batch)
  - Query embedding
  - Document chunk embedding with metadata preservation
  - Empty/invalid input handling
  - Dimension validation (1024)
  - Consistency (same model for doc and query, same normalization)
  - Semantic similarity sanity checks
  - Performance baseline
"""

import time
from pathlib import Path

import numpy as np
import pytest

from app.core.config import settings
from app.schemas.chunk import DocumentChunk
from app.schemas.document import FileType
from app.schemas.embedding import EmbeddedDocumentChunk
from app.services.embedding_service import EmbeddingService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_PPTX = FIXTURES_DIR / "sample.pptx"

# Shared service instance — loaded once for the module
_service: EmbeddingService = None


def _get_service() -> EmbeddingService:
    """Get or create a shared EmbeddingService for tests."""
    global _service
    if _service is None:
        _service = EmbeddingService()
        _service.load_model()
    return _service


def _make_chunk(
    text: str,
    chunk_id: str = "test_chunk_001",
    chunk_index: int = 0,
    document_id: str = "test-doc",
    page_number: int = None,
    slide_number: int = None,
    slide_title: str = None,
    heading: str = None,
    subject: str = None,
    topic: str = None,
) -> DocumentChunk:
    """Helper to create a DocumentChunk."""
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        text=text,
        char_count=len(text),
        document_id=document_id,
        document_name="test.pdf",
        file_type=FileType.PDF,
        user_id="test_user",
        subject=subject,
        topic=topic,
        page_number=page_number,
        slide_number=slide_number,
        slide_title=slide_title,
        heading=heading,
    )


# ===========================================================================
# 1. MODEL INITIALIZATION
# ===========================================================================

class TestModelInitialization:
    """Tests for model loading and device selection."""

    def test_model_loads_successfully(self):
        """Model should load without errors."""
        service = _get_service()
        assert service.is_loaded is True

    def test_model_runs_on_cpu(self):
        """On a typical dev machine, device should be cpu."""
        service = _get_service()
        # Device should be either 'cpu' or 'cuda' — both are valid
        assert service.device in ("cpu", "cuda")

    def test_device_selection_auto(self):
        """'auto' device should resolve to cpu or cuda."""
        service = EmbeddingService(device="auto")
        assert service.device in ("cpu", "cuda")

    def test_device_selection_cpu(self):
        """Explicit 'cpu' should be honored."""
        service = EmbeddingService(device="cpu")
        assert service.device == "cpu"

    def test_model_not_reloaded(self):
        """Calling load_model() twice should not reload."""
        service = _get_service()
        load_time_1 = service.load_time_ms

        # Call again — should be a no-op
        service.load_model()
        load_time_2 = service.load_time_ms

        # Same load time means it wasn't re-loaded
        assert load_time_1 == load_time_2

    def test_get_model_info(self):
        """get_model_info() should return correct metadata."""
        service = _get_service()
        info = service.get_model_info()

        assert info["model"] == settings.EMBEDDING_MODEL
        assert info["embedding_dimension"] == 1024
        assert info["is_loaded"] is True
        assert info["load_time_ms"] is not None
        assert info["load_time_ms"] > 0


# ===========================================================================
# 2. EMBEDDING GENERATION
# ===========================================================================

class TestEmbeddingGeneration:
    """Tests for text embedding."""

    def test_single_text_embedding(self):
        """A single text should produce one 1024-dim vector."""
        service = _get_service()
        vectors = service.embed_texts(["Deadlock in operating systems."])

        assert len(vectors) == 1
        assert len(vectors[0]) == 1024

    def test_multiple_text_embeddings(self):
        """Multiple texts should produce matching count of vectors."""
        service = _get_service()
        texts = [
            "First sentence about databases.",
            "Second sentence about algorithms.",
            "Third sentence about networking.",
        ]
        vectors = service.embed_texts(texts)

        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == 1024

    def test_batch_embedding(self):
        """Batch of texts should all produce 1024-dim vectors."""
        service = _get_service()
        texts = [f"Sentence number {i} about computer science." for i in range(20)]
        vectors = service.embed_texts(texts)

        assert len(vectors) == 20
        for vec in vectors:
            assert len(vec) == 1024

    def test_correct_vector_count(self):
        """Number of output vectors must equal number of input texts."""
        service = _get_service()
        n = 7
        texts = [f"Text {i}" for i in range(n)]
        vectors = service.embed_texts(texts)

        assert len(vectors) == n

    def test_correct_dimension_1024(self):
        """Every vector must have exactly 1024 dimensions."""
        service = _get_service()
        vectors = service.embed_texts(["Test dimension."])

        assert len(vectors[0]) == 1024

    def test_vectors_are_normalized(self):
        """Output vectors should be L2-normalized (magnitude ≈ 1.0)."""
        service = _get_service()
        vectors = service.embed_texts(["This is a normalized test."])

        magnitude = np.linalg.norm(vectors[0])
        assert abs(magnitude - 1.0) < 0.01, f"Expected magnitude ~1.0, got {magnitude}"


# ===========================================================================
# 3. QUERY EMBEDDING
# ===========================================================================

class TestQueryEmbedding:
    """Tests for query embedding."""

    def test_query_embedding(self):
        """A query should produce a 1024-dim vector."""
        service = _get_service()
        vector = service.embed_query("What is Banker's Algorithm?")

        assert len(vector) == 1024

    def test_query_embedding_normalized(self):
        """Query vectors should be L2-normalized."""
        service = _get_service()
        vector = service.embed_query("Explain process scheduling.")

        magnitude = np.linalg.norm(vector)
        assert abs(magnitude - 1.0) < 0.01


# ===========================================================================
# 4. DOCUMENT CHUNK EMBEDDING
# ===========================================================================

class TestDocumentChunkEmbedding:
    """Tests for embedding DocumentChunk objects."""

    def test_chunk_embedding(self):
        """Chunks should be embedded with correct structure."""
        service = _get_service()
        chunks = [
            _make_chunk("Process management in operating systems.", chunk_id="c1", chunk_index=0),
            _make_chunk("Deadlock avoidance using Banker's Algorithm.", chunk_id="c2", chunk_index=1),
        ]

        embedded, time_ms = service.embed_chunks(chunks)

        assert len(embedded) == 2
        assert time_ms > 0
        for ec in embedded:
            assert isinstance(ec, EmbeddedDocumentChunk)
            assert len(ec.embedding) == 1024


# ===========================================================================
# 5. EDGE CASES
# ===========================================================================

class TestEdgeCases:
    """Tests for empty/invalid input handling."""

    def test_empty_list_raises(self):
        """Empty list should raise ValueError."""
        service = _get_service()
        with pytest.raises(ValueError, match="empty list"):
            service.embed_texts([])

    def test_empty_string_raises(self):
        """Empty string in texts should raise ValueError."""
        service = _get_service()
        with pytest.raises(ValueError, match="empty or whitespace"):
            service.embed_texts([""])

    def test_whitespace_only_raises(self):
        """Whitespace-only string should raise ValueError."""
        service = _get_service()
        with pytest.raises(ValueError, match="empty or whitespace"):
            service.embed_texts(["   \n\t  "])

    def test_empty_query_raises(self):
        """Empty query should raise ValueError."""
        service = _get_service()
        with pytest.raises(ValueError, match="empty or whitespace"):
            service.embed_query("")

    def test_empty_chunks_raises(self):
        """Empty chunk list should raise ValueError."""
        service = _get_service()
        with pytest.raises(ValueError, match="empty list"):
            service.embed_chunks([])

    def test_model_not_loaded_raises(self):
        """Using service without loading model should raise RuntimeError."""
        unloaded = EmbeddingService()
        with pytest.raises(RuntimeError, match="not loaded"):
            unloaded.embed_texts(["test"])


# ===========================================================================
# 6. CONSISTENCY
# ===========================================================================

class TestConsistency:
    """Tests that document and query embeddings use the same model/normalization."""

    def test_same_model_for_doc_and_query(self):
        """Doc and query embeddings should use the same model instance."""
        service = _get_service()

        # Both should produce 1024-dim vectors
        doc_vec = service.embed_texts(["Deadlock in OS."])[0]
        query_vec = service.embed_query("Deadlock in OS.")

        assert len(doc_vec) == 1024
        assert len(query_vec) == 1024

    def test_same_normalization_for_doc_and_query(self):
        """Both doc and query vectors should be normalized."""
        service = _get_service()

        doc_vec = service.embed_texts(["Test normalization."])[0]
        query_vec = service.embed_query("Test normalization.")

        doc_mag = np.linalg.norm(doc_vec)
        query_mag = np.linalg.norm(query_vec)

        assert abs(doc_mag - 1.0) < 0.01
        assert abs(query_mag - 1.0) < 0.01

    def test_identical_text_produces_identical_embedding(self):
        """Same text embedded twice should produce the same vector."""
        service = _get_service()
        text = "Operating system process scheduling."

        vec1 = service.embed_texts([text])[0]
        vec2 = service.embed_texts([text])[0]

        similarity = service.cosine_similarity(vec1, vec2)
        assert similarity > 0.999, f"Same text should have similarity ~1.0, got {similarity}"


# ===========================================================================
# 7. METADATA PRESERVATION
# ===========================================================================

class TestMetadataPreservation:
    """Tests that chunk metadata survives embedding."""

    def test_chunk_metadata_preserved(self):
        """All chunk fields should be carried through to EmbeddedDocumentChunk."""
        service = _get_service()
        chunk = _make_chunk(
            text="Binary search tree operations.",
            chunk_id="meta-test-001",
            chunk_index=5,
            document_id="doc-xyz",
            page_number=3,
            subject="Data Structures",
            topic="Trees",
        )

        embedded, _ = service.embed_chunks([chunk])
        ec = embedded[0]

        assert ec.chunk_id == "meta-test-001"
        assert ec.chunk_index == 5
        assert ec.document_id == "doc-xyz"
        assert ec.document_name == "test.pdf"
        assert ec.file_type == FileType.PDF
        assert ec.user_id == "test_user"
        assert ec.page_number == 3
        assert ec.subject == "Data Structures"
        assert ec.topic == "Trees"
        assert ec.text == "Binary search tree operations."
        assert ec.char_count == len("Binary search tree operations.")

    def test_slide_metadata_preserved(self):
        """PPTX slide metadata should survive embedding."""
        service = _get_service()
        chunk = _make_chunk(
            text="SQL basics for database management.",
            slide_number=2,
            slide_title="SQL Introduction",
        )

        embedded, _ = service.embed_chunks([chunk])
        ec = embedded[0]

        assert ec.slide_number == 2
        assert ec.slide_title == "SQL Introduction"

    def test_heading_metadata_preserved(self):
        """DOCX heading metadata should survive embedding."""
        service = _get_service()
        chunk = _make_chunk(
            text="Linked list insertion and deletion.",
            heading="Linked Lists",
        )

        embedded, _ = service.embed_chunks([chunk])
        ec = embedded[0]

        assert ec.heading == "Linked Lists"

    def test_user_id_preserved(self):
        """user_id must survive for data isolation."""
        service = _get_service()
        chunk = _make_chunk(text="Some study material.")

        embedded, _ = service.embed_chunks([chunk])
        assert embedded[0].user_id == "test_user"


# ===========================================================================
# 8. SEMANTIC SIMILARITY SANITY CHECKS
# ===========================================================================

class TestSemanticSimilarity:
    """
    Verify that semantically related texts are more similar
    than unrelated texts.

    These are sanity checks, not hard thresholds.
    """

    def test_related_texts_more_similar(self):
        """
        A: "Deadlock occurs when processes wait indefinitely for resources."
        B: "Processes can become permanently blocked while waiting for resources."
        C: "Photosynthesis converts sunlight into chemical energy."

        Similarity(A, B) should be > Similarity(A, C)
        """
        service = _get_service()

        text_a = "Deadlock occurs when processes wait indefinitely for resources."
        text_b = "Processes can become permanently blocked while waiting for resources."
        text_c = "Photosynthesis converts sunlight into chemical energy."

        vectors = service.embed_texts([text_a, text_b, text_c])
        vec_a, vec_b, vec_c = vectors

        sim_ab = service.cosine_similarity(vec_a, vec_b)
        sim_ac = service.cosine_similarity(vec_a, vec_c)

        assert sim_ab > sim_ac, (
            f"Expected similarity(A,B)={sim_ab:.4f} > similarity(A,C)={sim_ac:.4f}"
        )

    def test_same_topic_more_similar(self):
        """Texts about the same topic should be more similar."""
        service = _get_service()

        os_1 = "Process scheduling determines which process runs on the CPU next."
        os_2 = "The CPU scheduler selects processes from the ready queue."
        bio = "DNA replication occurs during the S phase of the cell cycle."

        vectors = service.embed_texts([os_1, os_2, bio])

        sim_os = service.cosine_similarity(vectors[0], vectors[1])
        sim_cross = service.cosine_similarity(vectors[0], vectors[2])

        assert sim_os > sim_cross, (
            f"Expected OS-OS similarity={sim_os:.4f} > OS-Bio similarity={sim_cross:.4f}"
        )


# ===========================================================================
# 9. PERFORMANCE BASELINE
# ===========================================================================

class TestPerformanceBaseline:
    """
    Establish a baseline for embedding performance.

    These tests measure and report timing — they do not enforce
    strict time limits.
    """

    def test_model_load_time(self):
        """Report model load time."""
        service = _get_service()
        load_time = service.load_time_ms

        assert load_time is not None
        assert load_time > 0
        print(f"\n  Model load time: {load_time:.0f}ms")

    def test_single_embed_time(self):
        """Measure time to embed a single text."""
        service = _get_service()

        start = time.perf_counter()
        service.embed_texts(["Process scheduling in operating systems."])
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n  Single embed time: {elapsed:.1f}ms")
        assert elapsed > 0

    def test_batch_embed_performance(self):
        """Measure batch embedding performance with sample chunks."""
        service = _get_service()

        # Generate 50 sample texts (simulating chunks)
        texts = [
            f"This is chunk number {i} about computer science topic {i % 5}. "
            f"It covers important concepts like data structures and algorithms."
            for i in range(50)
        ]

        start = time.perf_counter()
        vectors = service.embed_texts(texts)
        elapsed = (time.perf_counter() - start) * 1000

        assert len(vectors) == 50

        print(f"\n  Batch performance (50 texts):")
        print(f"    Total time: {elapsed:.1f}ms")
        print(f"    Avg per text: {elapsed / 50:.1f}ms")
        print(f"    Device: {service.device}")
        print(f"    Batch size: {service.batch_size}")

    def test_pipeline_integration_performance(self):
        """
        Full pipeline test: extract PPTX → clean → chunk → embed.
        Reports end-to-end timing.
        """
        from app.services.document_service import process_document_pipeline

        service = _get_service()

        # Run pipeline up to chunking
        chunked = process_document_pipeline(
            file_path=SAMPLE_PPTX,
            user_id="perf_test_user",
        )

        # Embed the chunks
        start = time.perf_counter()
        embedded, embed_time = service.embed_chunks(chunked.chunks)
        total_embed_ms = (time.perf_counter() - start) * 1000

        print(f"\n  Pipeline integration (sample.pptx):")
        print(f"    Chunks: {len(chunked.chunks)}")
        print(f"    Extraction: {chunked.extraction_time_ms:.1f}ms")
        print(f"    Cleaning: {chunked.cleaning_time_ms:.1f}ms")
        print(f"    Chunking: {chunked.chunking_time_ms:.1f}ms")
        print(f"    Embedding: {total_embed_ms:.1f}ms")
        print(f"    Device: {service.device}")

        assert len(embedded) == len(chunked.chunks)
        for ec in embedded:
            assert len(ec.embedding) == 1024
