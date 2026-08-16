"""
BGE-M3 Embedding Service.

Loads the BAAI/bge-m3 model once and provides methods for:
  - embed_texts()  — batch embed raw text strings
  - embed_query()  — embed a single user query
  - embed_chunks() — embed DocumentChunk objects with metadata preservation

All embeddings are L2-normalized for cosine similarity compatibility.
Both document and query embeddings use the same model and normalization
strategy to ensure consistent retrieval.

Model produces 1024-dimensional vectors.
"""

import time
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.chunk import DocumentChunk
from app.schemas.embedding import EmbeddedDocumentChunk

logger = get_logger(__name__)


class EmbeddingService:
    """
    Manages the BGE-M3 embedding model lifecycle and provides
    embedding methods for documents and queries.

    Usage:
        service = EmbeddingService()
        service.load_model()
        vectors = service.embed_texts(["some text"])
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        expected_dimension: Optional[int] = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.expected_dimension = expected_dimension or settings.EMBEDDING_DIMENSION
        self._model: Optional[SentenceTransformer] = None
        self._model_loaded = False
        self._load_time_ms: Optional[float] = None

        # --- Resolve device ---
        requested_device = device or settings.EMBEDDING_DEVICE
        if requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested_device

        logger.info(
            "EmbeddingService initialized: model=%s  device=%s  batch_size=%d  dim=%d",
            self.model_name,
            self.device,
            self.batch_size,
            self.expected_dimension,
        )

    @property
    def dimension(self) -> int:
        """Return expected vector dimension."""
        return self.expected_dimension

    @property
    def embedding_dimension(self) -> int:
        """Return expected vector dimension."""
        return self.expected_dimension

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model_loaded

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Download (first time) and load the embedding model.

        This should be called once at application startup.
        Subsequent calls are no-ops.
        """
        if self._model_loaded:
            logger.debug("Model already loaded — skipping reload")
            return

        logger.info("Loading embedding model: %s ...", self.model_name)
        start = time.perf_counter()

        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            self._model_loaded = True
            self._load_time_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Embedding model loaded: model=%s  device=%s  load_time=%.1fms",
                self.model_name,
                self.device,
                self._load_time_ms,
            )
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model_loaded

    @property
    def load_time_ms(self) -> Optional[float]:
        """Time taken to load the model in milliseconds."""
        return self._load_time_ms

    def _ensure_loaded(self) -> None:
        """Raise if the model hasn't been loaded yet."""
        if not self._model_loaded or self._model is None:
            raise RuntimeError(
                "Embedding model not loaded. Call load_model() first."
            )

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.

        Parameters
        ----------
        texts : list[str]
            Texts to embed. Must not be empty.
            Each text must contain non-whitespace characters.

        Returns
        -------
        list[list[float]]
            List of 1024-dimensional normalized vectors.

        Raises
        ------
        ValueError
            If texts is empty or contains empty/whitespace-only strings.
        RuntimeError
            If model is not loaded or dimension mismatch.
        """
        self._ensure_loaded()

        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        # Validate individual texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"Text at index {i} is empty or whitespace-only. "
                    "Cannot generate meaningful embeddings for empty text."
                )

        logger.info(
            "Embedding %d text(s): batch_size=%d  device=%s",
            len(texts),
            self.batch_size,
            self.device,
        )
        start = time.perf_counter()

        # Encode with normalization for cosine similarity
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Validate dimension
        if embeddings.shape[1] != self.expected_dimension:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {self.expected_dimension}, "
                f"got {embeddings.shape[1]}. Model may have changed."
            )

        logger.info(
            "Embedding complete: count=%d  dim=%d  time=%.1fms  avg=%.1fms/text",
            len(embeddings),
            embeddings.shape[1],
            elapsed_ms,
            elapsed_ms / len(texts),
        )

        # Convert numpy arrays to Python lists
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Uses the same model and normalization as document embeddings
        to ensure consistent similarity computation.

        Parameters
        ----------
        query : str
            The user query text.

        Returns
        -------
        list[float]
            1024-dimensional normalized vector.
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty or whitespace-only.")

        vectors = self.embed_texts([query])
        return vectors[0]

    # ------------------------------------------------------------------
    # Document chunk embedding
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> tuple[list[EmbeddedDocumentChunk], float]:
        """
        Embed a list of DocumentChunks, preserving all metadata.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Chunks to embed. Must not be empty.

        Returns
        -------
        tuple[list[EmbeddedDocumentChunk], float]
            List of embedded chunks and the embedding time in ms.
        """
        if not chunks:
            raise ValueError("Cannot embed an empty list of chunks.")

        start = time.perf_counter()

        # Extract text from each chunk
        texts = [chunk.text for chunk in chunks]

        # Batch embed all texts
        vectors = self.embed_texts(texts)

        # Pair each chunk with its embedding
        embedded_chunks = [
            EmbeddedDocumentChunk.from_chunk_and_vector(chunk, vector)
            for chunk, vector in zip(chunks, vectors)
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Chunk embedding complete: chunks=%d  time=%.1fms",
            len(embedded_chunks),
            elapsed_ms,
        )

        return embedded_chunks, round(elapsed_ms, 2)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """Return model metadata for diagnostics."""
        return {
            "model": self.model_name,
            "device": self.device,
            "embedding_dimension": self.expected_dimension,
            "batch_size": self.batch_size,
            "is_loaded": self._model_loaded,
            "load_time_ms": self._load_time_ms,
        }

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)
