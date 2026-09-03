"""Anthropic adapter (Messages API).

No audio support here — the registry routes transcription to OpenAI or Gemini
and only then calls this provider with the resulting text.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from anthropic import APIError, AsyncAnthropic

from app.llm.base import (
    ChatTurn,
    LLMProvider,
    LLMReply,
    ProviderError,
    normalize_turns,
)
from app.llm.mime import clean, is_image, is_pdf

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supported_media = frozenset({"image", "document"})

    def __init__(self, api_key: str, default_model: str, timeout: float = 90.0) -> None:
        super().__init__(api_key, default_model, timeout)
        self._client: AsyncAnthropic | None = None

    @property
    def client(self) -> AsyncAnthropic:
        """Lazily built client, so an unconfigured provider costs nothing."""
        self.require_key()
        if self._client is None:
            self._client = AsyncAnthropic(
                api_key=self.api_key, timeout=self.timeout, max_retries=2
            )
        return self._client

    def _content(self, turn: ChatTurn) -> Any:
        """Turn -> Anthropic content blocks.

        Attachments come first: Claude follows instructions about a document
        better when the document precedes the question.
        """
        if turn.role == "assistant" or not turn.attachments:
            return turn.text

        blocks: list[dict[str, Any]] = []
        for item in turn.attachments:
            if item.kind == "image" and is_image(item.mime_type):
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": clean(item.mime_type),
                            "data": item.b64,
                        },
                    }
                )
            elif item.kind == "document" and is_pdf(item.mime_type):
                blocks.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": item.b64,
                        },
                    }
                )
            else:
                logger.info(
                    "Anthropic: skipping unsupported attachment %s (%s)",
                    item.kind,
                    item.mime_type,
                )

        if turn.text.strip():
            blocks.append({"type": "text", "text": turn.text})
        return blocks or turn.text

    async def chat(
        self,
        turns: list[ChatTurn],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.6,
    ) -> LLMReply:
        messages = [
            {"role": turn.role, "content": self._content(turn)}
            for turn in normalize_turns(turns)
        ]
        if not messages:
            raise ProviderError("No user message to send.")

        # Note: `system` is a top-level argument here, not a message role.
        params: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            params["system"] = system

        started = perf_counter()
        try:
            response = await self.client.messages.create(**params)
        except APIError as exc:
            raise ProviderError(f"Anthropic failed: {exc}") from exc

        # The answer arrives as a list of blocks; keep the text ones.
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)

        return LLMReply(
            text=text.strip(),
            provider=self.name,
            model=getattr(response, "model", params["model"]),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            latency_ms=int((perf_counter() - started) * 1000),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
