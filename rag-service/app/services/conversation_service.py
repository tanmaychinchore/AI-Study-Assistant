"""
Conversation & Message Storage Service using MongoDB.

Manages persistent multi-turn conversations and message histories with:
1. Strict tenant/user isolation on every query, write, and delete
2. Cascade message cleanup on conversation deletion
3. Chronological message retrieval (oldest to newest)
4. Configurable recent message history budgeting
5. Sanitized error handling and logging (no credentials or URI leakage)
"""

from datetime import datetime, timezone
import time
from typing import Any, Optional
import uuid

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.conversation import (
    ConversationResponse,
    MessageResponse,
    MessageRole,
)

logger = get_logger(__name__)

DEFAULT_CONVERSATION_TITLE = "New Study Conversation"


class ConversationServiceError(Exception):
    """Base exception for conversation operations."""
    pass


class ConversationNotFoundError(ConversationServiceError):
    """Raised when a requested conversation is not found or does not belong to the user."""
    pass


class ConversationAccessDeniedError(ConversationServiceError):
    """Raised when access to a conversation is rejected due to tenant mismatch."""
    pass


class ConversationService:
    """
    Service wrapper for MongoDB conversation and message persistence.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        database_name: Optional[str] = None,
        conversations_collection: Optional[str] = None,
        messages_collection: Optional[str] = None,
        client: Optional[MongoClient] = None,
    ) -> None:
        self.uri = uri or settings.MONGODB_URI
        self.database_name = database_name or settings.MONGODB_DATABASE
        self.conversations_collection_name = (
            conversations_collection or settings.MONGODB_CONVERSATIONS_COLLECTION
        )
        self.messages_collection_name = (
            messages_collection or settings.MONGODB_MESSAGES_COLLECTION
        )

        self._client: Optional[MongoClient] = client
        self._db: Optional[Database] = None
        self._conversations: Optional[Collection] = None
        self._messages: Optional[Collection] = None
        self._is_connected = False

        if client is not None:
            self._db = client[self.database_name]
            self._conversations = self._db[self.conversations_collection_name]
            self._messages = self._db[self.messages_collection_name]
            self._is_connected = True
            self._create_indexes()

        logger.info(
            "ConversationService initialized: db='%s'  conversations='%s'  messages='%s'",
            self.database_name,
            self.conversations_collection_name,
            self.messages_collection_name,
        )

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        """
        Establish connection to MongoDB and ensure indexes exist.
        """
        if self._is_connected and self._client is not None:
            return

        try:
            logger.info("Connecting to MongoDB database '%s'...", self.database_name)
            self._client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=2500,
                connectTimeoutMS=2500,
            )
            # Verify connectivity with ping
            self._client.admin.command("ping")
            self._db = self._client[self.database_name]
            self._conversations = self._db[self.conversations_collection_name]
            self._messages = self._db[self.messages_collection_name]
            self._is_connected = True

            self._create_indexes()
            logger.info("MongoDB connection established successfully.")

        except (ServerSelectionTimeoutError, ConnectionFailure) as exc:
            self._is_connected = False
            logger.warning("Failed to connect to MongoDB: %s", exc)
            raise ConversationServiceError(
                "Unable to connect to MongoDB conversation store."
            ) from exc
        except Exception as exc:
            self._is_connected = False
            logger.error("Unexpected error connecting to MongoDB: %s", exc)
            raise ConversationServiceError(f"MongoDB connection error: {exc}") from exc

    def _create_indexes(self) -> None:
        """Create optimal query indexes for conversations and messages."""
        if self._conversations is None or self._messages is None:
            return

        try:
            # 1. User conversations lookup (user_id + updated_at DESC)
            self._conversations.create_index(
                [("user_id", pymongo.ASCENDING), ("updated_at", pymongo.DESCENDING)],
                name="idx_user_updated",
            )
            # 2. Conversation isolation lookup (conversation_id + user_id UNIQUE)
            self._conversations.create_index(
                [("conversation_id", pymongo.ASCENDING), ("user_id", pymongo.ASCENDING)],
                unique=True,
                name="idx_conv_user_unique",
            )
            # 3. Message chronological ordering (conversation_id + created_at ASC)
            self._messages.create_index(
                [("conversation_id", pymongo.ASCENDING), ("created_at", pymongo.ASCENDING)],
                name="idx_msg_conv_created",
            )
            # 4. Message tenant isolation (conversation_id + user_id)
            self._messages.create_index(
                [("conversation_id", pymongo.ASCENDING), ("user_id", pymongo.ASCENDING)],
                name="idx_msg_conv_user",
            )
            logger.info("MongoDB indexes verified for conversations and messages.")
        except Exception as exc:
            logger.warning("Index creation encountered warning: %s", exc)

    def close(self) -> None:
        """Cleanly close the MongoDB client."""
        if self._client is not None:
            try:
                self._client.close()
                logger.info("MongoDB client connection closed.")
            except Exception as exc:
                logger.warning("Error closing MongoDB client: %s", exc)
            finally:
                self._is_connected = False
                self._client = None
                self._db = None
                self._conversations = None
                self._messages = None

    def _ensure_connected(self) -> None:
        """Validate connection status before executing database commands."""
        if not self._is_connected or self._conversations is None or self._messages is None:
            # Attempt to auto-connect
            self.connect()

    # -----------------------------------------------------------------------
    # Conversation CRUD Operations
    # -----------------------------------------------------------------------

    def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
    ) -> ConversationResponse:
        """
        Create and persist a new conversation session.
        """
        self._ensure_connected()

        if not user_id or not user_id.strip():
            raise ValueError("user_id is required to create a conversation.")

        clean_user_id = user_id.strip()
        clean_title = (title.strip() if title and title.strip() else DEFAULT_CONVERSATION_TITLE)
        conversation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        doc = {
            "conversation_id": conversation_id,
            "user_id": clean_user_id,
            "title": clean_title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

        try:
            self._conversations.insert_one(doc)
            logger.info(
                "Created conversation '%s' for user '%s' (title='%s')",
                conversation_id,
                clean_user_id,
                clean_title,
            )
            return ConversationResponse(
                conversation_id=conversation_id,
                user_id=clean_user_id,
                title=clean_title,
                created_at=now,
                updated_at=now,
                message_count=0,
            )
        except PyMongoError as exc:
            logger.error("MongoDB error creating conversation: %s", exc)
            raise ConversationServiceError(f"Failed to create conversation: {exc}") from exc

    def get_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> ConversationResponse:
        """
        Retrieve a conversation verifying exact user_id ownership.
        """
        self._ensure_connected()

        clean_conv_id = conversation_id.strip()
        clean_user_id = user_id.strip()

        try:
            doc = self._conversations.find_one(
                {"conversation_id": clean_conv_id, "user_id": clean_user_id}
            )
            if not doc:
                logger.warning(
                    "Conversation '%s' not found for user '%s'",
                    clean_conv_id,
                    clean_user_id,
                )
                raise ConversationNotFoundError(
                    f"Conversation '{clean_conv_id}' not found or access denied."
                )

            return ConversationResponse(
                conversation_id=doc["conversation_id"],
                user_id=doc["user_id"],
                title=doc["title"],
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
                message_count=doc.get("message_count", 0),
            )
        except PyMongoError as exc:
            logger.error("MongoDB error retrieving conversation: %s", exc)
            raise ConversationServiceError(f"Database error fetching conversation: {exc}") from exc

    def list_conversations(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ConversationResponse], int]:
        """
        List conversations for a specific user ordered by updated_at descending.
        """
        self._ensure_connected()

        clean_user_id = user_id.strip()

        try:
            filter_query = {"user_id": clean_user_id}
            total = self._conversations.count_documents(filter_query)

            cursor = (
                self._conversations.find(filter_query)
                .sort("updated_at", pymongo.DESCENDING)
                .skip(offset)
                .limit(limit)
            )

            conversations = [
                ConversationResponse(
                    conversation_id=doc["conversation_id"],
                    user_id=doc["user_id"],
                    title=doc["title"],
                    created_at=doc["created_at"],
                    updated_at=doc["updated_at"],
                    message_count=doc.get("message_count", 0),
                )
                for doc in cursor
            ]

            return conversations, total
        except PyMongoError as exc:
            logger.error("MongoDB error listing conversations: %s", exc)
            raise ConversationServiceError(f"Database error listing conversations: {exc}") from exc

    def delete_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete a conversation and cascade delete all associated messages.
        """
        self._ensure_connected()

        clean_conv_id = conversation_id.strip()
        clean_user_id = user_id.strip()

        # Enforce user ownership check
        self.get_conversation(clean_conv_id, clean_user_id)

        try:
            # 1. Delete conversation document
            res_conv = self._conversations.delete_one(
                {"conversation_id": clean_conv_id, "user_id": clean_user_id}
            )

            # 2. Cascade delete messages
            res_msg = self._messages.delete_many(
                {"conversation_id": clean_conv_id, "user_id": clean_user_id}
            )

            logger.info(
                "Deleted conversation '%s' for user '%s' (cascade removed %d messages)",
                clean_conv_id,
                clean_user_id,
                res_msg.deleted_count,
            )
            return res_conv.deleted_count > 0

        except PyMongoError as exc:
            logger.error("MongoDB error deleting conversation: %s", exc)
            raise ConversationServiceError(f"Database error deleting conversation: {exc}") from exc

    # -----------------------------------------------------------------------
    # Message CRUD Operations
    # -----------------------------------------------------------------------

    def append_message(
        self,
        conversation_id: str,
        user_id: str,
        role: MessageRole,
        content: str,
    ) -> MessageResponse:
        """
        Append a new message to a conversation and update timestamps & message count.
        """
        self._ensure_connected()

        clean_conv_id = conversation_id.strip()
        clean_user_id = user_id.strip()
        clean_content = content.strip()

        if not clean_content:
            raise ValueError("Message content cannot be empty.")

        # Ensure conversation exists and belongs to user
        self.get_conversation(clean_conv_id, clean_user_id)

        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        msg_doc = {
            "message_id": message_id,
            "conversation_id": clean_conv_id,
            "user_id": clean_user_id,
            "role": role.value if isinstance(role, MessageRole) else str(role),
            "content": clean_content,
            "created_at": now,
        }

        try:
            self._messages.insert_one(msg_doc)

            # Atomically increment count and update timestamp
            self._conversations.update_one(
                {"conversation_id": clean_conv_id, "user_id": clean_user_id},
                {
                    "$set": {"updated_at": now},
                    "$inc": {"message_count": 1},
                },
            )

            logger.info(
                "Appended %s message '%s' to conversation '%s'",
                role.value if isinstance(role, MessageRole) else str(role),
                message_id,
                clean_conv_id,
            )

            return MessageResponse(
                message_id=message_id,
                conversation_id=clean_conv_id,
                user_id=clean_user_id,
                role=role if isinstance(role, MessageRole) else MessageRole(role),
                content=clean_content,
                created_at=now,
            )
        except PyMongoError as exc:
            logger.error("MongoDB error appending message: %s", exc)
            raise ConversationServiceError(f"Database error appending message: {exc}") from exc

    def get_messages(
        self,
        conversation_id: str,
        user_id: str,
        limit: Optional[int] = None,
    ) -> list[MessageResponse]:
        """
        Retrieve all messages in a conversation in chronological order (oldest to newest).
        """
        self._ensure_connected()

        clean_conv_id = conversation_id.strip()
        clean_user_id = user_id.strip()

        # Verify ownership
        self.get_conversation(clean_conv_id, clean_user_id)

        try:
            cursor = self._messages.find(
                {"conversation_id": clean_conv_id, "user_id": clean_user_id}
            ).sort("created_at", pymongo.ASCENDING)

            if limit is not None and limit > 0:
                cursor = cursor.limit(limit)

            return [
                MessageResponse(
                    message_id=doc["message_id"],
                    conversation_id=doc["conversation_id"],
                    user_id=doc["user_id"],
                    role=MessageRole(doc["role"]),
                    content=doc["content"],
                    created_at=doc["created_at"],
                )
                for doc in cursor
            ]
        except PyMongoError as exc:
            logger.error("MongoDB error fetching messages: %s", exc)
            raise ConversationServiceError(f"Database error fetching messages: {exc}") from exc

    def get_recent_history(
        self,
        conversation_id: str,
        user_id: str,
        limit: int = 10,
    ) -> list[MessageResponse]:
        """
        Retrieve the latest N messages from a conversation in chronological order (oldest to newest).
        """
        self._ensure_connected()

        clean_conv_id = conversation_id.strip()
        clean_user_id = user_id.strip()

        # Verify ownership
        self.get_conversation(clean_conv_id, clean_user_id)

        try:
            # Fetch latest N messages (newest first), then reverse to restore oldest -> newest order
            cursor = (
                self._messages.find(
                    {"conversation_id": clean_conv_id, "user_id": clean_user_id}
                )
                .sort("created_at", pymongo.DESCENDING)
                .limit(limit)
            )

            recent_docs = list(cursor)
            recent_docs.reverse()

            return [
                MessageResponse(
                    message_id=doc["message_id"],
                    conversation_id=doc["conversation_id"],
                    user_id=doc["user_id"],
                    role=MessageRole(doc["role"]),
                    content=doc["content"],
                    created_at=doc["created_at"],
                )
                for doc in recent_docs
            ]
        except PyMongoError as exc:
            logger.error("MongoDB error fetching recent history: %s", exc)
            raise ConversationServiceError(f"Database error fetching history: {exc}") from exc

    def get_health_info(self) -> dict[str, Any]:
        """Return connectivity status for health endpoints without leaking credentials."""
        if not self._is_connected:
            return {
                "status": "unconfigured",
                "database": self.database_name,
                "connected": False,
            }
        try:
            self._client.admin.command("ping")
            return {
                "status": "connected",
                "database": self.database_name,
                "conversations_collection": self.conversations_collection_name,
                "messages_collection": self.messages_collection_name,
                "connected": True,
            }
        except Exception as exc:
            return {
                "status": f"unhealthy ({exc})",
                "database": self.database_name,
                "connected": False,
            }
