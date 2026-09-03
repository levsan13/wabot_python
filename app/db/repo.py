"""Data access — small functions built on top of an AsyncSession.

Keeping the queries here (instead of inline in the services) makes the handler
readable and lets the tests exercise persistence without HTTP.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message, ProcessedEvent


# ----------------------------------------------------------------- conversations
async def get_or_create_conversation(
    session: AsyncSession, wa_id: str, display_name: str | None = None
) -> Conversation:
    """Fetch the thread for a number, creating it on first contact."""
    result = await session.execute(select(Conversation).where(Conversation.wa_id == wa_id))
    conversation = result.scalar_one_or_none()

    if conversation is None:
        conversation = Conversation(wa_id=wa_id, display_name=display_name)
        session.add(conversation)
        try:
            await session.flush()
        except IntegrityError:
            # Two webhooks for the same new number raced us; reuse the winner.
            await session.rollback()
            result = await session.execute(
                select(Conversation).where(Conversation.wa_id == wa_id)
            )
            conversation = result.scalar_one()
    elif display_name and conversation.display_name != display_name:
        # People rename their WhatsApp profile; keep the latest.
        conversation.display_name = display_name

    return conversation


async def list_conversations(
    session: AsyncSession, limit: int = 50, offset: int = 0
) -> list[Conversation]:
    """Most recently active threads first."""
    result = await session.execute(
        select(Conversation)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def reset_conversation(session: AsyncSession, conversation: Conversation) -> int:
    """Drop the history and the summary. Returns how many messages were removed."""
    count = await session.scalar(
        select(func.count(Message.id)).where(Message.conversation_id == conversation.id)
    )
    await session.execute(delete(Message).where(Message.conversation_id == conversation.id))
    conversation.summary = None
    conversation.summarized_until_id = 0
    conversation.message_count = 0
    return int(count or 0)


# ---------------------------------------------------------------------- messages
async def add_message(
    session: AsyncSession,
    conversation: Conversation,
    role: str,
    content: str,
    *,
    media_kind: str | None = None,
    media_mime: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    wa_message_id: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: int | None = None,
) -> Message:
    """Append a turn and bump the conversation counters."""
    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=content,
        media_kind=media_kind,
        media_mime=media_mime,
        provider=provider,
        model=model,
        wa_message_id=wa_message_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )
    session.add(message)
    conversation.message_count = (conversation.message_count or 0) + 1
    conversation.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return message


async def recent_messages(
    session: AsyncSession, conversation_id: int, limit: int = 20, after_id: int = 0
) -> list[Message]:
    """Last `limit` messages, returned in chronological order.

    `after_id` skips everything already folded into the summary.
    """
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.id > after_id)
        .order_by(Message.id.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))


async def messages_between(
    session: AsyncSession, conversation_id: int, after_id: int, before_id: int
) -> list[Message]:
    """Half-open range (after_id, before_id] — the slice the summarizer compresses."""
    result = await session.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.id > after_id,
            Message.id <= before_id,
        )
        .order_by(Message.id)
    )
    return list(result.scalars().all())


async def count_messages(session: AsyncSession, conversation_id: int, after_id: int = 0) -> int:
    total = await session.scalar(
        select(func.count(Message.id)).where(
            Message.conversation_id == conversation_id, Message.id > after_id
        )
    )
    return int(total or 0)


# ------------------------------------------------------------------ deduplication
async def claim_event(session: AsyncSession, wa_message_id: str) -> bool:
    """True on the first sighting of an event, False when it is a redelivery."""
    exists = await session.get(ProcessedEvent, wa_message_id)
    if exists is not None:
        return False
    session.add(ProcessedEvent(wa_message_id=wa_message_id, epoch=int(time.time())))
    try:
        await session.flush()
    except IntegrityError:
        # Another worker claimed the same event microseconds earlier.
        await session.rollback()
        return False
    return True


async def purge_old_events(session: AsyncSession, days: int = 3) -> int:
    """Housekeeping: Meta never retries an event this old."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await session.execute(
        delete(ProcessedEvent).where(ProcessedEvent.received_at < cutoff)
    )
    return int(result.rowcount or 0)
