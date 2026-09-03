"""Provider registry: picks who answers and falls back when they fail."""

from __future__ import annotations

import logging

from app.config import Settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import (
    Attachment,
    ChatTurn,
    LLMProvider,
    LLMReply,
    ProviderError,
    ProviderNotConfigured,
)
from app.llm.gemini_provider import GeminiProvider
from app.llm.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Holds one instance of each provider and decides which one to use."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # All three are always instantiated; clients are only built on first use,
        # so providers without an API key cost nothing.
        self._providers: dict[str, LLMProvider] = {
            "openai": OpenAIProvider(
                api_key=settings.openai_api_key,
                default_model=settings.openai_model,
                timeout=settings.request_timeout,
                transcribe_model=settings.openai_transcribe_model,
                base_url=settings.openai_base_url,
            ),
            "anthropic": AnthropicProvider(
                api_key=settings.anthropic_api_key,
                default_model=settings.anthropic_model,
                timeout=settings.request_timeout,
            ),
            "gemini": GeminiProvider(
                api_key=settings.gemini_api_key,
                default_model=settings.gemini_model,
                timeout=settings.request_timeout,
                transcribe_model=settings.gemini_transcribe_model,
            ),
        }

    # -------------------------------------------------------------- queries
    def get(self, name: str) -> LLMProvider:
        provider = self._providers.get((name or "").lower())
        if provider is None:
            raise ProviderError(f"Unknown provider: {name!r}")
        return provider

    def all(self) -> dict[str, LLMProvider]:
        return dict(self._providers)

    def available(self) -> list[str]:
        """Providers that actually have an API key."""
        return [name for name, p in self._providers.items() if p.is_configured()]

    def is_available(self, name: str) -> bool:
        return (name or "").lower() in self.available()

    def resolve(self, preferred: str | None = None) -> str:
        """Pick a provider: the requested one, the default, or the first configured."""
        candidates = [preferred, self.settings.default_provider, *self.settings.fallback_order]
        available = self.available()
        for candidate in candidates:
            if candidate and candidate.lower() in available:
                return candidate.lower()
        if not available:
            raise ProviderNotConfigured(
                "No provider configured — set OPENAI_API_KEY, ANTHROPIC_API_KEY "
                "or GEMINI_API_KEY in .env."
            )
        return available[0]

    def order_for(self, preferred: str | None) -> list[str]:
        """Primary provider followed by the configured fallbacks."""
        primary = self.resolve(preferred)
        chain = [primary]
        for name in self.settings.fallback_order:
            if name not in chain and self.is_available(name):
                chain.append(name)
        return chain

    # ----------------------------------------------------------- generation
    async def chat(
        self,
        turns: list[ChatTurn],
        *,
        system: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        allow_fallback: bool = True,
    ) -> LLMReply:
        """Ask the chain until someone answers; raise if nobody does."""
        chain = self.order_for(provider) if allow_fallback else [self.resolve(provider)]
        max_tokens = max_tokens or self.settings.max_output_tokens
        temperature = self.settings.temperature if temperature is None else temperature

        errors: list[str] = []
        for index, name in enumerate(chain):
            engine = self.get(name)
            # A hand-picked model id only makes sense for the primary provider.
            chosen_model = model if index == 0 else None
            try:
                reply = await engine.chat(
                    turns,
                    system=system,
                    model=chosen_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if index > 0:
                    logger.warning("Answered by fallback provider '%s'", name)
                if reply.text:
                    return reply
                # An empty answer is a failure too: try the next provider.
                errors.append(f"{name}: empty response")
            except ProviderError as exc:
                logger.warning("Provider '%s' failed: %s", name, exc)
                errors.append(f"{name}: {exc}")

        raise ProviderError(" | ".join(errors) or "No provider answered.")

    # -------------------------------------------------------- transcription
    def transcriber(self) -> LLMProvider | None:
        """Provider used for speech to text, honouring TRANSCRIPTION_PROVIDER."""
        choice = self.settings.transcription_provider
        if choice == "none":
            return None
        provider = self._providers.get(choice)
        if provider and provider.can_transcribe():
            return provider
        # Configured choice is unusable: take any other capable provider.
        for candidate in self._providers.values():
            if candidate.can_transcribe():
                return candidate
        return None

    async def transcribe(self, audio: Attachment, language: str | None = "pt") -> str:
        provider = self.transcriber()
        if provider is None:
            raise ProviderError("No provider available to transcribe audio.")
        return await provider.transcribe(audio, language=language)

    async def aclose(self) -> None:
        """Best-effort shutdown of every provider's HTTP client."""
        for provider in self._providers.values():
            try:
                await provider.aclose()
            except Exception:  # pragma: no cover - best effort
                logger.debug("Failed to close provider %s", provider.name, exc_info=True)
