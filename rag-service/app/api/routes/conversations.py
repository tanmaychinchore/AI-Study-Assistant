"""
Conversations and multi-turn chat API routes.

Provides endpoints for:
1. Creating, retrieving, listing, and deleting conversations
2. Viewing persistent message histories
3. Executing conversation-aware multi-turn RAG chat sessions
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.api.dependencies.auth import get_current_user

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.conversation import (
    ChatRequest,
    ChatResponse,
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationResponse,
    MessageListResponse,
    MessageResponse,
    MessageRole,
)
from app.schemas.llm import ChatMessage
from app.schemas.rag import RAGRequest, RAGResult
from app.schemas.response import SuccessResponse
from app.services.conversation_service import (
    ConversationAccessDeniedError,
    ConversationNotFoundError,
    ConversationService,
    ConversationServiceError,
)
from app.services.groq_service import (
    GroqAuthError,
    GroqModelError,
    GroqRateLimitError,
    GroqServiceError,
    GroqTimeoutError,
)
from app.services.rag_service import RAGService

logger = get_logger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


def _get_conversation_service(request: Request) -> ConversationService:
    """Retrieve ConversationService from application state."""
    service = getattr(request.app.state, "conversation_service", None)
    if service is None or not service.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation storage service is not available on this server.",
        )
    return service


def _get_rag_service(request: Request) -> RAGService:
    """Retrieve RAGService from application state."""
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG generation service is not available on this server.",
        )
    return service


# ---------------------------------------------------------------------------
# 1. Conversation Lifecycle Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=SuccessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new study conversation",
    description="Initiates a persistent conversation session for a user.",
)
async def create_conversation_endpoint(
    request_body: ConversationCreateRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> SuccessResponse:
    if request_body.user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client user_id does not match the authenticated identity.",
        )
    conv_service = _get_conversation_service(request)

    try:
        conversation = conv_service.create_conversation(
            user_id=request_body.user_id,
            title=request_body.title,
        )
        return SuccessResponse(
            message="Conversation created successfully.",
            data=conversation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ConversationServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get(
    "",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="List a user's conversations",
    description="Returns all conversation sessions owned by the specified user_id.",
)
async def list_conversations_endpoint(
    request: Request,
    user_id: str = Query(..., min_length=1, description="User ID owning the conversations."),
    limit: int = Query(50, ge=1, le=100, description="Max conversations to return."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
    current_user: dict = Depends(get_current_user),
) -> SuccessResponse:
    if user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client user_id does not match the authenticated identity.",
        )
    conv_service = _get_conversation_service(request)

    try:
        conversations, total = conv_service.list_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
        return SuccessResponse(
            message=f"Retrieved {len(conversations)} conversation(s).",
            data=ConversationListResponse(conversations=conversations, total=total),
        )
    except ConversationServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get(
    "/{conversation_id}",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Get conversation details",
    description="Fetches a conversation by ID, verifying exact user ownership.",
)
async def get_conversation_endpoint(
    conversation_id: str,
    request: Request,
    user_id: str = Query(..., min_length=1, description="User ID requesting access."),
    current_user: dict = Depends(get_current_user),
) -> SuccessResponse:
    if user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client user_id does not match the authenticated identity.",
        )
    conv_service = _get_conversation_service(request)

    try:
        conversation = conv_service.get_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        return SuccessResponse(
            message="Conversation retrieved successfully.",
            data=conversation,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConversationAccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ConversationServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.delete(
    "/{conversation_id}",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete conversation",
    description="Deletes a conversation session and cascades deletion of all its messages.",
)
async def delete_conversation_endpoint(
    conversation_id: str,
    request: Request,
    user_id: str = Query(..., min_length=1, description="User ID owning the conversation."),
    current_user: dict = Depends(get_current_user),
) -> SuccessResponse:
    if user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client user_id does not match the authenticated identity.",
        )
    conv_service = _get_conversation_service(request)

    try:
        deleted = conv_service.delete_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        return SuccessResponse(
            message=f"Conversation '{conversation_id}' and all associated messages were deleted.",
            data={"conversation_id": conversation_id, "deleted": deleted},
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConversationAccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ConversationServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# 2. Message History Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{conversation_id}/messages",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Get conversation message history",
    description="Returns chronological message history (oldest to newest) for a conversation.",
)
async def get_messages_endpoint(
    conversation_id: str,
    request: Request,
    user_id: str = Query(..., min_length=1, description="User ID requesting message history."),
    limit: Optional[int] = Query(None, ge=1, le=500, description="Optional maximum message limit."),
    current_user: dict = Depends(get_current_user),
) -> SuccessResponse:
    if user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client user_id does not match the authenticated identity.",
        )
    conv_service = _get_conversation_service(request)

    try:
        messages = conv_service.get_messages(
            conversation_id=conversation_id,
            user_id=user_id,
            limit=limit,
        )
        return SuccessResponse(
            message=f"Retrieved {len(messages)} message(s).",
            data=MessageListResponse(
                conversation_id=conversation_id,
                messages=messages,
                total=len(messages),
            ),
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConversationAccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ConversationServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# 3. Multi-turn Conversational RAG Chat Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/{conversation_id}/chat",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message in a study conversation",
    description=(
        "Executes a conversation-aware multi-turn RAG turn: loads recent message history, "
        "contextualizes follow-up queries, searches indexed study chunks via BGE-M3 & Astra DB, "
        "generates a grounded answer with Groq LLM, and persists both user & assistant messages."
    ),
)
async def conversation_chat_endpoint(
    conversation_id: str,
    request_body: ChatRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> SuccessResponse:
    if request_body.user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client user_id does not match the authenticated identity.",
        )
    conv_service = _get_conversation_service(request)
    rag_service = _get_rag_service(request)

    # 1. Verify conversation ownership
    try:
        conv_service.get_conversation(
            conversation_id=conversation_id,
            user_id=request_body.user_id,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConversationAccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    # 2. Retrieve recent message history (bounded by CHAT_MAX_HISTORY_MESSAGES)
    recent_history_docs = conv_service.get_recent_history(
        conversation_id=conversation_id,
        user_id=request_body.user_id,
        limit=settings.CHAT_MAX_HISTORY_MESSAGES,
    )

    history_chat_messages = [
        ChatMessage(
            role="user" if msg.role == MessageRole.USER else "assistant",
            content=msg.content,
        )
        for msg in recent_history_docs
    ]

    # 3. Record user message in MongoDB before generation
    user_msg_response = conv_service.append_message(
        conversation_id=conversation_id,
        user_id=request_body.user_id,
        role=MessageRole.USER,
        content=request_body.message,
    )

    # 4. Execute conversation-aware RAG generation
    rag_request = RAGRequest(
        query=request_body.message,
        user_id=request_body.user_id,
        top_k=request_body.top_k,
        document_id=request_body.document_id,
        subject=request_body.subject,
        topic=request_body.topic,
        similarity_threshold=request_body.similarity_threshold,
    )

    try:
        rag_result: RAGResult = rag_service.query(
            request=rag_request,
            conversation_history=history_chat_messages,
        )

        # 5. On successful generation: persist assistant message
        asst_msg_response = conv_service.append_message(
            conversation_id=conversation_id,
            user_id=request_body.user_id,
            role=MessageRole.ASSISTANT,
            content=rag_result.answer,
        )

        chat_response = ChatResponse(
            conversation_id=conversation_id,
            user_message=user_msg_response,
            assistant_message=asst_msg_response,
            answer=rag_result.answer,
            grounded=rag_result.grounded,
            sources=rag_result.sources,
            retrieval_statistics=rag_result.retrieval_statistics,
            generation_statistics=rag_result.generation_statistics,
            total_time_ms=rag_result.total_time_ms,
        )

        return SuccessResponse(
            message="Chat response generated and saved successfully.",
            data=chat_response,
        )

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GroqAuthError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except GroqRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except GroqTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except GroqModelError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GroqServiceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error during conversation chat: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during chat: {exc}",
        ) from exc
