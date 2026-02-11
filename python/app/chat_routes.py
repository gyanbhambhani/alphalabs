"""Trading Lab Chat API endpoints"""
from typing import Optional, List as TypingList
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from app.chat import TradingLabChat

router = APIRouter(prefix="/api/lab", tags=["chat"])


class ChatQuery(BaseModel):
    """Chat query request"""
    query: str
    conversation_id: Optional[str] = None


class ChatMessageResponse(BaseModel):
    """Chat message response"""
    message: str
    query_type: str
    data: Optional[dict] = None
    suggestions: TypingList[str] = []
    sources: TypingList[str] = []


# Store chat instances per conversation
_chat_instances: dict = {}


def get_chat_instance(conversation_id: str = "default"):
    """Get or create chat instance for a conversation"""
    if conversation_id not in _chat_instances:
        _chat_instances[conversation_id] = TradingLabChat(
            persist_directory="./chroma_data"
        )
    return _chat_instances[conversation_id]


@router.post("/chat", response_model=ChatMessageResponse)
async def lab_chat(query: ChatQuery):
    """
    Handle conversational queries about markets.

    Examples:
    - "What happened after every Fed rate hike?"
    - "Find periods similar to current conditions"
    - "Compare AAPL and MSFT"
    - "Generate a research report on NVDA"
    """
    try:
        conversation_id = query.conversation_id or "default"
        chat = get_chat_instance(conversation_id)

        response = await chat.handle_query(query.query)

        return ChatMessageResponse(
            message=response.message,
            query_type=response.query_type.value,
            data=response.data,
            suggestions=response.suggestions,
            sources=response.sources
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {str(e)}"
        )


@router.get("/suggestions")
async def get_suggestions():
    """Get example queries for the chat"""
    chat = get_chat_instance()
    return {
        "suggestions": chat.example_queries
    }


@router.get("/history/{conversation_id}")
async def get_chat_history(conversation_id: str):
    """Get conversation history"""
    chat = get_chat_instance(conversation_id)
    return {
        "history": chat.get_conversation_history()
    }


@router.delete("/history/{conversation_id}")
async def clear_chat_history(conversation_id: str):
    """Clear conversation history"""
    chat = get_chat_instance(conversation_id)
    chat.clear_history()
    return {"status": "cleared"}
