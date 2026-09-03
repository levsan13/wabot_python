"""Google Gemini adapter (unified google-genai SDK).

The only one of the three that takes raw audio in a normal prompt, so it also
works as the transcriber when TRANSCRIPTION_PROVIDER=gemini.
"""

from __future__ import annotations

import logging
from time import perf_counter

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.llm.base import (
    Attachment,
    ChatTurn,
    LLMProvider,
    LLMReply,
    ProviderError,
    normalize_turns,
)
from app.llm.mime import clean

logger = logging.getLogger(__name__)

# Written in Portuguese because it steers the model transcribing pt-BR speech.
TRANSCRIBE_PROMPT = (
    "Transcreva o áudio a seguir literalmente, sem comentar, resumir ou traduzir. "
    "Responda apenas com o texto falado."
)


class GeminiProvider(LLMProvider):
    name = "gemini"
    supported_media = frozenset({"image", "document", "audio"})

    def __init__(
        self,
        api_key: str,
        default_model: str,
        timeout: float = 90.0,
        transcribe_model: str | None = None,
    ) -> None:
        super().__init__(api_key, default_model, timeout)
        self.transcribe_model = transcribe_model or default_model
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        """Lazily built client, so an unconfigured provider costs nothing."""
        self.require_key()
        if self._client is None:
            self._client = genai.Client(
                api_key=self.api_key,
                # google-genai takes the timeout in milliseconds.
                http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
            )
        return self._client

    @staticmethod
    def _parts(turn: ChatTurn) -> list[types.Part]:
        """Turn -> Gemini parts. Bytes go inline, no upload round-trip needed."""
        parts: list[types.Part] = []
        for item in turn.attachments:
            parts.append(types.Part.from_bytes(data=item.data, mime_type=clean(item.mime_type)))
        if turn.text.strip():
            parts.append(types.Part.from_text(text=turn.text))
        return parts

    async def chat(
        self,
        turns: list[ChatTurn],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.6,
    ) -> LLMReply:
        contents: list[types.Content] = []
        for turn in normalize_turns(turns):
            parts = self._parts(turn)
            if not parts:
                continue
            # Gemini calls the assistant role "model".
            contents.append(
                types.Content(role="user" if turn.role == "user" else "model", parts=parts)
            )
        if not contents:
            raise ProviderError("No user message to send.")

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        started = perf_counter()
        try:
            response = await self.client.aio.models.generate_content(
                model=model or self.default_model, contents=contents, config=config
            )
        except genai_errors.APIError as exc:
            raise ProviderError(f"Gemini failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        return LLMReply(
            text=(response.text or "").strip(),
            provider=self.name,
            model=model or self.default_model,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            latency_ms=int((perf_counter() - started) * 1000),
        )

    def can_transcribe(self) -> bool:
        return bool(self.api_key)

    async def transcribe(self, audio: Attachment, language: str | None = None) -> str:
        """Transcription is just a normal generation with an audio part."""
        prompt = TRANSCRIBE_PROMPT
        if language:
            prompt += f" O idioma falado é {language}."

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=audio.data, mime_type=clean(audio.mime_type)),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]
        try:
            response = await self.client.aio.models.generate_content(
                model=self.transcribe_model,
                contents=contents,
                # Temperature 0: transcription should not be creative.
                config=types.GenerateContentConfig(temperature=0.0),
            )
        except genai_errors.APIError as exc:
            raise ProviderError(f"Audio transcription failed (Gemini): {exc}") from exc
        return (response.text or "").strip()
