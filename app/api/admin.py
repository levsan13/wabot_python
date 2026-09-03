"""Support REST API: test the LLM, send a message, inspect conversations."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import AppContext, get_context, require_api_key
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationOut,
    MessageOut,
    SendMessageRequest,
    SendMessageResponse,
)
from app.db import repo
from app.db.base import get_session
from app.db.models import Conversation
from app.llm.base import ChatTurn, ProviderError
from app.whatsapp.client import WhatsAppError

# The API key guard applies to every route in this router.
router = APIRouter(prefix="/api", tags=["api"], dependencies=[Depends(require_api_key)])


@router.get("/providers")
async def providers(context: AppContext = Depends(get_context)) -> dict:
    """Which providers hold a key, and the model each one defaults to."""
    return {
        "default": context.settings.default_provider,
        "available": context.registry.available(),
        "models": {
            name: provider.default_model for name, provider in context.registry.all().items()
        },
        "transcriber": getattr(context.registry.transcriber(), "name", None),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, context: AppContext = Depends(get_context)) -> ChatResponse:
    """Talk to the LLM without WhatsApp — the quickest way to test routing."""
    turns = [ChatTurn(role=m.role, text=m.content) for m in payload.messages]
    try:
        reply = await context.registry.chat(
            turns,
            system=payload.system or context.settings.system_prompt,
            provider=payload.provider,
            model=payload.model,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            allow_fallback=payload.allow_fallback,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # LLMReply is a slots dataclass, so asdict() and not __dict__.
    return ChatResponse(**asdict(reply))


@router.post("/messages", response_model=SendMessageResponse)
async def send_message(
    payload: SendMessageRequest, context: AppContext = Depends(get_context)
) -> SendMessageResponse:
    """Send a text message from the bot's number.

    Subject to Meta's 24-hour customer service window: outside it only approved
    templates go through and this returns 502.
    """
    try:
        sent = await context.whatsapp.send_text(
            payload.to,
            payload.text,
            preview_url=payload.preview_url,
            reply_to=payload.reply_to,
        )
    except WhatsAppError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return SendMessageResponse(sent=sent)


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[ConversationOut]:
    rows = await repo.list_conversations(session, limit=limit, offset=offset)
    return [
        ConversationOut(
            wa_id=c.wa_id,
            display_name=c.display_name,
            provider=c.provider,
            model=c.model,
            message_count=c.message_count,
            has_summary=bool(c.summary),
            updated_at=c.updated_at,
        )
        for c in rows
    ]


async def _load_conversation(session: AsyncSession, wa_id: str) -> Conversation:
    """Fetch a thread by number or raise a clean 404."""
    result = await session.execute(select(Conversation).where(Conversation.wa_id == wa_id))
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation {wa_id} not found")
    return conversation


@router.get("/conversations/{wa_id}/messages", response_model=list[MessageOut])
async def conversation_messages(
    wa_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> list[MessageOut]:
    conversation = await _load_conversation(session, wa_id)
    messages = await repo.recent_messages(session, conversation.id, limit=limit)
    return [
        MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            provider=m.provider,
            model=m.model,
            media_kind=m.media_kind,
            input_tokens=m.input_tokens,
            output_tokens=m.output_tokens,
            created_at=m.created_at,
        )
        for m in messages
    ]


@router.delete("/conversations/{wa_id}/history")
async def clear_history(wa_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Same as the /reset command, but from the outside."""
    conversation = await _load_conversation(session, wa_id)
    removed = await repo.reset_conversation(session, conversation)
    return {"wa_id": wa_id, "removed": removed}
