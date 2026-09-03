"""Persisted models: conversations, messages and already-handled events."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(Base):
    """A one-to-one thread with a WhatsApp number."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    wa_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(128), default=None)

    # Per-conversation overrides, set through the /provedor and /modelo commands.
    provider: Mapped[str | None] = mapped_column(String(32), default=None)
    model: Mapped[str | None] = mapped_column(String(96), default=None)
    persona: Mapped[str | None] = mapped_column(Text, default=None)

    # Rolling summary of the turns that already fell out of the context window.
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    summarized_until_id: Mapped[int] = mapped_column(Integer, default=0)

    message_count: Mapped[int] = mapped_column(Integer, default=0)
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Conversation {self.wa_id} msgs={self.message_count}>"


class Message(Base):
    """One conversation turn, from either the user or the assistant."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conv_id", "conversation_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )

    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text, default="")

    # What arrived over WhatsApp; the bytes themselves are never stored.
    media_kind: Mapped[str | None] = mapped_column(String(16), default=None)
    media_mime: Mapped[str | None] = mapped_column(String(96), default=None)

    # Bookkeeping about the model that produced an assistant turn.
    provider: Mapped[str | None] = mapped_column(String(32), default=None)
    model: Mapped[str | None] = mapped_column(String(96), default=None)
    wa_message_id: Mapped[str | None] = mapped_column(String(128), default=None, index=True)

    input_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    output_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Message {self.role} {self.content[:32]!r}>"


class ProcessedEvent(Base):
    """Deduplication table: Meta redelivers a webhook that is not ack'ed fast."""

    __tablename__ = "processed_events"

    wa_message_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    epoch: Mapped[int] = mapped_column(BigInteger, default=0)
