"""OpenAI adapter — Chat Completions plus audio transcription.

Chat Completions (rather than the Responses API) because it is the most stable
surface across SDK versions; everything is funnelled through `chat()` so
swapping it later touches this file only.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from openai import APIError, AsyncOpenAI, BadRequestError

from app.llm.base import (
    Attachment,
    ChatTurn,
    LLMProvider,
    LLMReply,
    ProviderError,
    normalize_turns,
)
from app.llm.mime import guess_extension

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    name = "openai"
    supported_media = frozenset({"image", "document", "audio"})

    def __init__(
        self,
        api_key: str,
        default_model: str,
        timeout: float = 90.0,
        transcribe_model: str = "gpt-transcribe",
        base_url: str | None = None,
    ) -> None:
        super().__init__(api_key, default_model, timeout)
        self.transcribe_model = transcribe_model
        self._base_url = base_url or None
        self._client: AsyncOpenAI | None = None

    # ------------------------------------------------------------- internal
    @property
    def client(self) -> AsyncOpenAI:
        """Lazily built client, so an unconfigured provider costs nothing."""
        self.require_key()
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self._base_url,
                timeout=self.timeout,
                max_retries=2,
            )
        return self._client

    def _content(self, turn: ChatTurn) -> Any:
        """Turn -> OpenAI message content (plain string or a list of parts)."""
        # The assistant role only accepts text.
        if turn.role == "assistant" or not turn.attachments:
            return turn.text

        parts: list[dict[str, Any]] = []
        if turn.text.strip():
            parts.append({"type": "text", "text": turn.text})

        for item in turn.attachments:
            if item.kind == "image":
                parts.append({"type": "image_url", "image_url": {"url": item.data_uri}})
            elif item.kind == "document":
                parts.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": item.filename or "arquivo.pdf",
                            "file_data": item.data_uri,
                        },
                    }
                )
            else:
                # Audio is transcribed upstream; anything left is just flagged.
                parts.append({"type": "text", "text": "[áudio não transcrito]"})

        return parts or turn.text

    @staticmethod
    def _relax(params: dict[str, Any], error: Exception) -> dict[str, Any] | None:
        """Build a retry without the parameter the model rejected.

        Reasoning models refuse `temperature`, and some deployments still want
        the legacy `max_tokens`. Returns None when the error is something else.
        """
        message = str(error).lower()
        if "temperature" in message and "temperature" in params:
            retry = dict(params)
            retry.pop("temperature")
            return retry
        if "max_completion_tokens" in message and "max_completion_tokens" in params:
            retry = dict(params)
            retry["max_tokens"] = retry.pop("max_completion_tokens")
            return retry
        return None

    # ------------------------------------------------------------------ API
    async def chat(
        self,
        turns: list[ChatTurn],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.6,
    ) -> LLMReply:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for turn in normalize_turns(turns):
            messages.append({"role": turn.role, "content": self._content(turn)})

        if not messages or messages[-1]["role"] == "system":
            raise ProviderError("No user message to send.")

        params: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }

        started = perf_counter()
        try:
            response = await self.client.chat.completions.create(**params)
        except BadRequestError as exc:
            retry = self._relax(params, exc)
            if retry is None:
                raise ProviderError(f"OpenAI rejected the request: {exc}") from exc
            logger.warning("OpenAI: retrying without unsupported parameter (%s)", exc)
            try:
                response = await self.client.chat.completions.create(**retry)
            except APIError as exc2:  # pragma: no cover - depends on the live API
                raise ProviderError(f"OpenAI failed: {exc2}") from exc2
        except APIError as exc:
            raise ProviderError(f"OpenAI failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""
        usage = getattr(response, "usage", None)

        return LLMReply(
            text=text.strip(),
            provider=self.name,
            model=getattr(response, "model", params["model"]),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=int((perf_counter() - started) * 1000),
        )

    def can_transcribe(self) -> bool:
        return bool(self.api_key and self.transcribe_model)

    async def transcribe(self, audio: Attachment, language: str | None = None) -> str:
        """Voice note -> text. The filename matters: the API sniffs the format."""
        filename = audio.filename or f"audio{guess_extension(audio.mime_type, '.ogg')}"
        kwargs: dict[str, Any] = {
            "model": self.transcribe_model,
            "file": (filename, audio.data, audio.mime_type.split(";")[0].strip()),
        }
        if language:
            kwargs["language"] = language
        try:
            result = await self.client.audio.transcriptions.create(**kwargs)
        except APIError as exc:
            raise ProviderError(f"Audio transcription failed (OpenAI): {exc}") from exc

        # Depending on response_format the SDK returns an object or a bare string.
        text = getattr(result, "text", None)
        return (text if isinstance(text, str) else str(result)).strip()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
