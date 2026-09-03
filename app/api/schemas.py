"""Request and response models of the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str


class ChatRequest(BaseModel):
    """Talk to the LLM directly, bypassing WhatsApp."""

    messages: list[ChatMessageIn] = Field(min_length=1)
    provider: Literal["openai", "anthropic", "gemini"] | None = None
    model: str | None = None
    system: str | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=32000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    allow_fallback: bool = True


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


class SendMessageRequest(BaseModel):
    to: str = Field(description="E.164 number without '+', e.g. 5585999998888")
    text: str = Field(min_length=1)
    preview_url: bool = False
    reply_to: str | None = Field(default=None, description="wamid to quote in the reply")


class SendMessageResponse(BaseModel):
    sent: list[str] = Field(description="wamids created (more than one if split)")


class ConversationOut(BaseModel):
    wa_id: str
    display_name: str | None = None
    provider: str | None = None
    model: str | None = None
    message_count: int = 0
    has_summary: bool = False
    updated_at: datetime | None = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    provider: str | None = None
    model: str | None = None
    media_kind: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    created_at: datetime | None = None
