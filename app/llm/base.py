"""The contract every LLM provider implements.

The rest of the application only ever deals with `ChatTurn`, `Attachment` and
`LLMReply`; each adapter translates those into its own SDK's shape. Adding a
fourth provider means writing one file, not touching the handler.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

MediaKind = Literal["image", "document", "audio"]
Role = Literal["user", "assistant"]


class ProviderError(RuntimeError):
    """Provider call failed (network, quota, bad key, ...). Triggers the fallback."""


class ProviderNotConfigured(ProviderError):
    """No API key for this provider."""


class CapabilityNotSupported(ProviderError):
    """Provider cannot do what was asked (e.g. Anthropic does not take audio)."""


@dataclass(slots=True)
class Attachment:
    """A file already downloaded from WhatsApp, held in memory."""

    kind: MediaKind
    mime_type: str
    data: bytes
    filename: str | None = None

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def data_uri(self) -> str:
        """`data:` URI, which is how OpenAI takes inline files."""
        return f"data:{self.mime_type};base64,{self.b64}"

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass(slots=True)
class ChatTurn:
    """One turn of the conversation, optionally carrying attachments."""

    role: Role
    text: str = ""
    attachments: list[Attachment] = field(default_factory=list)


@dataclass(slots=True)
class LLMReply:
    """Normalized answer, whichever provider produced it."""

    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


def normalize_turns(turns: list[ChatTurn]) -> list[ChatTurn]:
    """Drop empty turns, remove leading assistant turns and merge repeated roles.

    Anthropic and Gemini require strict user/assistant alternation starting with
    the user; OpenAI is lenient. Normalizing in one place keeps the three
    adapters free of that bookkeeping.
    """
    cleaned = [t for t in turns if t.text.strip() or t.attachments]
    while cleaned and cleaned[0].role != "user":
        cleaned.pop(0)

    merged: list[ChatTurn] = []
    for turn in cleaned:
        if merged and merged[-1].role == turn.role:
            previous = merged[-1]
            joined = "\n\n".join(p for p in (previous.text, turn.text) if p.strip())
            merged[-1] = ChatTurn(
                role=previous.role,
                text=joined,
                attachments=[*previous.attachments, *turn.attachments],
            )
        else:
            merged.append(turn)
    return merged


class LLMProvider(ABC):
    """Interface implemented by the OpenAI, Anthropic and Gemini adapters."""

    name: str = "base"
    supported_media: frozenset[str] = frozenset()

    def __init__(self, api_key: str, default_model: str, timeout: float = 90.0) -> None:
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout

    # -------------------------------------------------------------- helpers
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def require_key(self) -> None:
        """Guard called before touching the SDK client."""
        if not self.is_configured():
            raise ProviderNotConfigured(f"Provider '{self.name}' has no API key configured.")

    def supports(self, kind: MediaKind) -> bool:
        return kind in self.supported_media

    def can_transcribe(self) -> bool:
        return False

    # ------------------------------------------------------------------ API
    @abstractmethod
    async def chat(
        self,
        turns: list[ChatTurn],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.6,
    ) -> LLMReply:
        """Produce the assistant's answer for the given conversation."""

    async def transcribe(self, audio: Attachment, language: str | None = None) -> str:
        """Speech to text. Overridden by the providers that support it."""
        raise CapabilityNotSupported(f"'{self.name}' cannot transcribe audio.")

    async def aclose(self) -> None:  # pragma: no cover - most SDKs need nothing
        """Release HTTP resources at shutdown."""
        return None
