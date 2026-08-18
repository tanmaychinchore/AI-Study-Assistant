"""
Astra DB Vector Storage Service.

Manages connection to Astra DB Serverless, creates and validates the
vector-enabled collection (`study_chunks`), and provides operations for:
  - insert_embedded_chunks() — batch store EmbeddedDocumentChunk objects
  - get_chunk()              — fetch a stored chunk by ID
  - delete_chunk()           — delete a single chunk by ID
  - delete_document_chunks() — delete all chunks belonging to a document
  - get_health()             — report connection and collection health

Vector configuration:
  - Dimension : 1024 (matching BGE-M3 output)
  - Metric    : cosine (VectorMetric.COSINE)
"""

from datetime import datetime
import time
from typing import Any, Optional

from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.database import Database
from astrapy.collection import Collection

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.embedding import EmbeddedDocumentChunk

logger = get_logger(__name__)


class AstraDBService:
    """
    Service wrapper around Astra DB Serverless for vector storage and retrieval.

    Handles connection lifecycle, collection validation (1024-dim, cosine),
    batch insertion, and ID-based retrieval.
    """

    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        token: Optional[str] = None,
        keyspace: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.api_endpoint = (
            api_endpoint if api_endpoint is not None else settings.ASTRA_DB_API_ENDPOINT
        ).strip()
        self.token = (
            token if token is not None else settings.ASTRA_DB_APPLICATION_TOKEN
        ).strip()
        self.keyspace = (
            keyspace if keyspace is not None else settings.ASTRA_DB_KEYSPACE
        ).strip()
        self.collection_name = (
            collection_name if collection_name is not None else settings.ASTRA_DB_COLLECTION_NAME
        ).strip()
        self.expected_dimension = (
            dimension if dimension is not None else settings.EMBEDDING_DIMENSION
        )
        self.metric = "cosine"

        self._client: Optional[DataAPIClient] = None
        self._database: Optional[Database] = None
        self._collection: Optional[Collection] = None
        self._is_connected = False
        self._collection_ready = False

    @property
    def is_configured(self) -> bool:
        """Check if required Astra DB credentials are provided."""
        return bool(self.api_endpoint and self.token)

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to Astra DB."""
        return self._is_connected

    @property
    def is_ready(self) -> bool:
        """Check if database is connected and collection is verified."""
        return self._is_connected and self._collection_ready

    # ------------------------------------------------------------------
    # Connection Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Initialize the DataAPIClient and connect to the Astra DB database.
        """
        if not self.is_configured:
            logger.warning("Astra DB credentials not configured in settings. Skipping connection.")
            return

        logger.info(
            "Connecting to Astra DB: endpoint=%s  keyspace=%s  collection=%s",
            self._mask_endpoint(self.api_endpoint),
            self.keyspace,
            self.collection_name,
        )

        try:
            self._client = DataAPIClient(token=self.token)
            self._database = self._client.get_database(
                api_endpoint=self.api_endpoint,
                keyspace=self.keyspace,
            )
            # Test connectivity by pinging/listing collections
            self._database.list_collection_names()
            self._is_connected = True
            logger.info("Astra DB connection established successfully.")
        except Exception as exc:
            self._is_connected = False
            logger.error("Failed to connect to Astra DB: %s", exc)
            raise ConnectionError(f"Failed to connect to Astra DB: {exc}") from exc

    def initialize_collection(self) -> None:
        """
        Ensure the target collection exists with 1024 dimensions and cosine metric.

        If the collection does not exist, creates it.
        If it exists, validates that dimension and metric match expected values.
        """
        if not self._is_connected or self._database is None:
            raise ConnectionError("Cannot initialize collection: Astra DB is not connected.")

        logger.info(
            "Checking Astra DB collection: '%s' (dimension=%d, metric=%s)...",
            self.collection_name,
            self.expected_dimension,
            self.metric,
        )

        try:
            existing_collections = self._database.list_collection_names()

            if self.collection_name not in existing_collections:
                logger.info("Collection '%s' does not exist. Creating new vector collection...", self.collection_name)
                self._collection = self._database.create_collection(
                    name=self.collection_name,
                    dimension=self.expected_dimension,
                    metric=VectorMetric.COSINE,
                    check_exists=False,
                )
                logger.info("Vector collection '%s' created successfully.", self.collection_name)
            else:
                logger.info("Collection '%s' already exists. Validating configuration...", self.collection_name)
                self._collection = self._database.get_collection(self.collection_name)

                # Validate collection vector configuration
                self._validate_existing_collection_options()

            self._collection_ready = True
            logger.info("Collection '%s' is ready for operations.", self.collection_name)

        except Exception as exc:
            self._collection_ready = False
            logger.error("Failed to initialize collection '%s': %s", self.collection_name, exc)
            raise

    def _validate_existing_collection_options(self) -> None:
        """Inspect and validate vector dimension and metric of an existing collection."""
        try:
            # Look up collection descriptor from db.list_collections()
            collection_descriptors = self._database.list_collections()
            descriptor = next(
                (c for c in collection_descriptors if c.name == self.collection_name),
                None,
            )

            if descriptor and descriptor.options and descriptor.options.vector:
                vector_opts = descriptor.options.vector
                dim = vector_opts.dimension
                metric = vector_opts.metric

                if dim != self.expected_dimension:
                    raise RuntimeError(
                        f"Collection '{self.collection_name}' has vector dimension {dim}, "
                        f"expected {self.expected_dimension}. Please use a collection with dimension {self.expected_dimension}."
                    )
                if metric and metric.lower() != self.metric.lower():
                    raise RuntimeError(
                        f"Collection '{self.collection_name}' has metric '{metric}', "
                        f"expected '{self.metric}'. Please use a collection configured with cosine metric."
                    )
                logger.info("Collection options verified: dimension=%d, metric=%s", dim, metric)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("Could not inspect collection options (skipping strict check): %s", exc)

    # ------------------------------------------------------------------
    # Vector Storage Operations
    # ------------------------------------------------------------------

    def insert_embedded_chunks(
        self,
        chunks: list[EmbeddedDocumentChunk],
        batch_size: int = 25,
    ) -> tuple[int, list[str], float]:
        """
        Batch insert embedded chunks into Astra DB.

        Parameters
        ----------
        chunks : list[EmbeddedDocumentChunk]
            List of chunks with 1024-dim vectors and metadata.
        batch_size : int
            Number of documents to send per batch insert request (default: 25).

        Returns
        -------
        tuple[int, list[str], float]
            (inserted_count, inserted_ids, duration_ms)
        """
        if not chunks:
            raise ValueError("Cannot insert empty list of chunks.")

        if not self._collection_ready or self._collection is None:
            raise RuntimeError("Astra DB collection is not initialized. Cannot insert chunks.")

        start_time = time.perf_counter()
        inserted_ids: list[str] = []
        total_inserted = 0

        # Transform chunks to Astra DB document format
        documents = [self._chunk_to_document(c) for c in chunks]

        logger.info(
            "Inserting %d chunk(s) into Astra DB collection '%s' (batch_size=%d)...",
            len(documents),
            self.collection_name,
            batch_size,
        )

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                # Use insert_many with ordered=False for best throughput
                result = self._collection.insert_many(
                    documents=batch,
                    ordered=False,
                )
                batch_ids = list(result.inserted_ids)
                inserted_ids.extend(batch_ids)
                total_inserted += len(batch_ids)
                logger.debug(
                    "Inserted batch %d/%d (%d chunks)",
                    (i // batch_size) + 1,
                    (len(documents) + batch_size - 1) // batch_size,
                    len(batch_ids),
                )
            except Exception as exc:
                logger.error("Error inserting batch at index %d: %s", i, exc)
                raise RuntimeError(f"Failed to insert chunks into Astra DB: {exc}") from exc

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Astra DB insert complete: count=%d  time=%.1fms  avg=%.1fms/chunk",
            total_inserted,
            duration_ms,
            duration_ms / total_inserted if total_inserted else 0,
        )

        return total_inserted, inserted_ids, round(duration_ms, 2)

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """
        Retrieve a chunk document by its chunk_id (`_id`).

        Returns None if not found.
        """
        if not self._collection_ready or self._collection is None:
            raise RuntimeError("Astra DB collection is not initialized.")

        doc = self._collection.find_one(
            filter={"_id": chunk_id},
            projection={"*": True},
        )
        if not doc:
            return None

        # Verify vector presence and dimension
        has_vector = "$vector" in doc and isinstance(doc["$vector"], list)
        vec_dim = len(doc["$vector"]) if has_vector else 0

        # Return standardized structure
        return {
            "chunk_id": doc.get("chunk_id", doc.get("_id")),
            "document_id": doc.get("document_id", ""),
            "document_name": doc.get("document_name", ""),
            "user_id": doc.get("user_id", ""),
            "text": doc.get("text", ""),
            "char_count": doc.get("char_count", len(doc.get("text", ""))),
            "file_type": doc.get("file_type", "unknown"),
            "page_number": doc.get("page_number"),
            "slide_number": doc.get("slide_number"),
            "slide_title": doc.get("slide_title"),
            "heading": doc.get("heading"),
            "subject": doc.get("subject"),
            "topic": doc.get("topic"),
            "chunk_index": doc.get("chunk_index", 0),
            "source_type": doc.get("source_type", "document"),
            "has_vector": has_vector,
            "vector_dimension": vec_dim,
            "created_at": doc.get("created_at"),
        }

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a single chunk by its ID.

        Returns True if deleted, False otherwise.
        """
        if not self._collection_ready or self._collection is None:
            raise RuntimeError("Astra DB collection is not initialized.")

        result = self._collection.delete_one({"_id": chunk_id})
        return result.deleted_count > 0

    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks associated with a given document_id.

        Returns the number of deleted chunks.
        """
        if not self._collection_ready or self._collection is None:
            raise RuntimeError("Astra DB collection is not initialized.")

        result = self._collection.delete_many({"document_id": document_id})
        logger.info("Deleted %d chunk(s) for document_id='%s'", result.deleted_count, document_id)
        return result.deleted_count

    # ------------------------------------------------------------------
    # Vector Search & Retrieval Operations
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query_vector: list[float],
        user_id: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> tuple[list[dict], float, int]:
        """
        Perform vector similarity search against Astra DB with metadata filtering.

        Parameters
        ----------
        query_vector : list[float]
            1024-dimensional normalized query embedding.
        user_id : str
            Owner user ID (strictly enforced for user isolation).
        top_k : int
            Number of nearest chunks to retrieve.
        document_id : Optional[str]
            Filter by specific document ID.
        subject : Optional[str]
            Filter by subject tag.
        topic : Optional[str]
            Filter by topic tag.
        similarity_threshold : Optional[float]
            Minimum cosine similarity score cutoff.

        Returns
        -------
        tuple[list[dict], float, int]
            (ranked_chunks_dicts, search_time_ms, raw_chunks_retrieved_count)
        """
        if not self._collection_ready or self._collection is None:
            raise RuntimeError("Astra DB collection is not initialized.")

        if not query_vector:
            raise ValueError("Query vector cannot be empty.")

        if len(query_vector) != self.expected_dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match "
                f"expected dimension {self.expected_dimension}."
            )

        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty or whitespace-only.")

        # Build filter criteria enforcing user isolation
        filter_dict: dict[str, Any] = {"user_id": user_id.strip()}
        if document_id is not None and document_id.strip():
            filter_dict["document_id"] = document_id.strip()
        if subject is not None and subject.strip():
            filter_dict["subject"] = subject.strip()
        if topic is not None and topic.strip():
            filter_dict["topic"] = topic.strip()

        logger.info(
            "Executing Astra DB vector search: user_id='%s'  top_k=%d  filters=%s",
            user_id,
            top_k,
            filter_dict,
        )

        # Explicit projection excludes large $vector embeddings to save network bandwidth
        projection = {
            "_id": True,
            "chunk_id": True,
            "document_id": True,
            "document_name": True,
            "user_id": True,
            "text": True,
            "char_count": True,
            "file_type": True,
            "page_number": True,
            "slide_number": True,
            "slide_title": True,
            "heading": True,
            "subject": True,
            "topic": True,
            "chunk_index": True,
            "source_type": True,
        }

        start_time = time.perf_counter()

        # Step 1: Astra DB retrieves top_k nearest candidate chunks
        cursor = self._collection.find(
            filter=filter_dict,
            sort={"$vector": query_vector},
            limit=top_k,
            include_similarity=True,
            projection=projection,
        )

        raw_docs = list(cursor)
        search_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        raw_count = len(raw_docs)

        logger.info(
            "Astra DB search complete: retrieved=%d  time=%.1fms",
            raw_count,
            search_time_ms,
        )

        # Step 2: Post-filtering — similarity threshold is applied after the top-K candidate retrieval
        results: list[dict] = []
        for doc in raw_docs:
            raw_sim = doc.get("$similarity", 0.0)
            sim_score = float(raw_sim) if raw_sim is not None else 0.0
            if similarity_threshold is not None and sim_score < similarity_threshold:
                logger.debug(
                    "Skipping candidate chunk '%s' (similarity %.4f < threshold %.4f)",
                    doc.get("_id"),
                    sim_score,
                    similarity_threshold,
                )
                continue

            results.append({
                "chunk_id": doc.get("chunk_id", doc.get("_id")),
                "document_id": doc.get("document_id", ""),
                "document_name": doc.get("document_name", ""),
                "user_id": doc.get("user_id", ""),
                "text": doc.get("text", ""),
                "char_count": doc.get("char_count", len(doc.get("text", ""))),
                "file_type": doc.get("file_type", "unknown"),
                "page_number": doc.get("page_number"),
                "slide_number": doc.get("slide_number"),
                "slide_title": doc.get("slide_title"),
                "heading": doc.get("heading"),
                "subject": doc.get("subject"),
                "topic": doc.get("topic"),
                "chunk_index": doc.get("chunk_index", 0),
                "source_type": doc.get("source_type", "document"),
                "similarity_score": round(sim_score, 4),
            })

        return results, search_time_ms, raw_count

    # ------------------------------------------------------------------
    # Health & Diagnostics
    # ------------------------------------------------------------------

    def get_health(self) -> dict:
        """
        Return diagnostic health information about Astra DB.
        """
        if not self.is_configured:
            return {
                "status": "not_configured",
                "keyspace": self.keyspace,
                "collection": self.collection_name,
                "vector_dimension": self.expected_dimension,
                "metric": self.metric,
                "is_connected": False,
                "collection_exists": False,
                "detail": "Astra DB credentials are not set in .env",
            }

        if not self._is_connected:
            return {
                "status": "disconnected",
                "keyspace": self.keyspace,
                "collection": self.collection_name,
                "vector_dimension": self.expected_dimension,
                "metric": self.metric,
                "is_connected": False,
                "collection_exists": False,
                "detail": "Failed to connect to Astra DB endpoint.",
            }

        return {
            "status": "connected" if self._collection_ready else "collection_not_ready",
            "keyspace": self.keyspace,
            "collection": self.collection_name,
            "vector_dimension": self.expected_dimension,
            "metric": self.metric,
            "is_connected": self._is_connected,
            "collection_exists": self._collection_ready,
            "detail": "Astra DB vector collection is ready for read/write operations.",
        }

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _chunk_to_document(self, chunk: EmbeddedDocumentChunk) -> dict[str, Any]:
        """Convert EmbeddedDocumentChunk into Astra DB document format."""
        file_type_val = chunk.file_type.value if hasattr(chunk.file_type, "value") else str(chunk.file_type)

        # Validate vector dimension before building doc
        if len(chunk.embedding) != self.expected_dimension:
            raise ValueError(
                f"Chunk '{chunk.chunk_id}' has vector dimension {len(chunk.embedding)}, "
                f"expected {self.expected_dimension}."
            )

        return {
            "_id": chunk.chunk_id,
            "$vector": chunk.embedding,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "user_id": chunk.user_id,
            "text": chunk.text,
            "char_count": chunk.char_count,
            "file_type": file_type_val,
            "page_number": chunk.page_number,
            "slide_number": chunk.slide_number,
            "slide_title": chunk.slide_title,
            "heading": chunk.heading,
            "subject": chunk.subject,
            "topic": chunk.topic,
            "chunk_index": chunk.chunk_index,
            "source_type": chunk.source_type,
            "created_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def _mask_endpoint(endpoint: str) -> str:
        """Mask endpoint for safe logging."""
        if not endpoint:
            return ""
        if len(endpoint) <= 25:
            return "***"
        return endpoint[:15] + "..." + endpoint[-10:]
