"""
Authentication Service managing password hashing, JWT encoding/decoding,
and user persistence in MongoDB.
"""

from datetime import datetime, timedelta
import uuid
from typing import Any, Optional

import jwt
from passlib.context import CryptContext
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Manages JWT generation/verification and user registration/login in MongoDB."""

    def __init__(
        self,
        uri: Optional[str] = None,
        database_name: Optional[str] = None,
        client: Optional[MongoClient] = None,
    ) -> None:
        self.uri = uri or settings.MONGODB_URI
        self.database_name = database_name or settings.MONGODB_DATABASE
        self._client = client
        self._db: Optional[Database] = None
        self._users: Optional[Collection] = None
        self._is_connected = False

    def connect(self) -> None:
        """Establish connection to MongoDB and ensure indexes exist."""
        if self._is_connected and self._client is not None:
            return

        try:
            if not self._client:
                self._client = MongoClient(
                    self.uri,
                    serverSelectionTimeoutMS=2500,
                    connectTimeoutMS=2500,
                )
            self._db = self._client[self.database_name]
            self._users = self._db["users"]
            
            # Ensure unique normalized email index
            self._users.create_index("normalized_email", unique=True)
            self._is_connected = True
            logger.info("AuthService MongoDB connection established and unique index verified.")
        except Exception as exc:
            logger.error("Failed to connect AuthService to MongoDB: %s", exc)
            self._is_connected = False
            raise

    def close(self) -> None:
        """Close connection to MongoDB."""
        if self._client and not self._client.nodes:
            try:
                self._client.close()
            except Exception:
                pass
        self._is_connected = False

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a plaintext password using bcrypt."""
        if not password or len(password) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a plaintext password against its bcrypt hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def register_user(self, email: str, password: str, name: Optional[str] = None) -> dict:
        """
        Register a new user in MongoDB.
        
        Normalizes the email address and checks for uniqueness.
        """
        if not self._is_connected:
            self.connect()

        normalized_email = email.strip().lower()
        if not normalized_email:
            raise ValueError("Email cannot be empty.")

        # Check if email is already taken
        existing = self._users.find_one({"normalized_email": normalized_email})
        if existing:
            raise ValueError("An account with this email already exists.")

        password_hash = self.hash_password(password)
        user_id = f"usr_{uuid.uuid4().hex}"
        now = datetime.utcnow()

        user_doc = {
            "_id": user_id,
            "user_id": user_id,
            "email": email.strip(),
            "normalized_email": normalized_email,
            "password_hash": password_hash,
            "name": name.strip() if name else None,
            "created_at": now,
            "updated_at": now,
        }

        self._users.insert_one(user_doc)
        logger.info("Successfully registered user ID: %s", user_id)
        
        # Return copy without password hash
        user_doc_copy = user_doc.copy()
        user_doc_copy.pop("password_hash", None)
        user_doc_copy.pop("normalized_email", None)
        return user_doc_copy

    def authenticate_user(self, email: str, password: str) -> dict:
        """
        Authenticate a user by email and password.
        
        Returns the user document (without password hash) on success, or raises ValueError.
        """
        if not self._is_connected:
            self.connect()

        normalized_email = email.strip().lower()
        user = self._users.find_one({"normalized_email": normalized_email})
        if not user:
            # Prevent timing differences and generic login error
            self.hash_password("dummy_long_password_to_prevent_timing_leaks")
            raise ValueError("Invalid email or password.")

        if not self.verify_password(password, user["password_hash"]):
            raise ValueError("Invalid email or password.")

        user_copy = user.copy()
        user_copy.pop("password_hash", None)
        user_copy.pop("normalized_email", None)
        return user_copy

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """Fetch a user by their user_id from MongoDB."""
        if not self._is_connected:
            self.connect()
        user = self._users.find_one({"_id": user_id})
        if user:
            user_copy = user.copy()
            user_copy.pop("password_hash", None)
            user_copy.pop("normalized_email", None)
            return user_copy
        return None

    @staticmethod
    def create_access_token(user_id: str) -> tuple[str, int]:
        """
        Generate a JWT access token for a given user_id.
        
        Returns a tuple of (token_string, expires_in_seconds).
        """
        expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        expires_delta = timedelta(minutes=expire_minutes)
        expire_time = datetime.utcnow() + expires_delta

        payload = {
            "sub": user_id,
            "exp": expire_time,
            "iat": datetime.utcnow(),
        }

        token = jwt.encode(
            payload,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )
        expires_in = int(expires_delta.total_seconds())
        return token, expires_in

    @staticmethod
    def decode_access_token(token: str) -> str:
        """
        Decode and validate a JWT access token.
        
        Returns the authenticated user_id (sub) from the token, or raises jwt.PyJWTError.
        """
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        user_id = payload.get("sub")
        if not user_id:
            raise jwt.PyJWTError("Token missing 'sub' claim.")
        return str(user_id)
