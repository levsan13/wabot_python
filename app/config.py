"""Central configuration — everything comes from env vars / the .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai", "anthropic", "gemini"]

# Kept in Portuguese on purpose: this is what the bot says to its users.
DEFAULT_SYSTEM_PROMPT = (
    "Você é um assistente que conversa pelo WhatsApp. "
    "Responda em português do Brasil, de forma direta e cordial. "
    "Prefira respostas curtas (até ~120 palavras), porque a pessoa lê no celular; "
    "só se estenda quando pedirem detalhes. "
    "Use apenas a formatação que o WhatsApp entende: *negrito*, _itálico_, ```código```. "
    "Nunca invente informação: se não souber, diga que não sabe."
)


class Settings(BaseSettings):
    """Every knob of the application, with sane defaults.

    Field names map to upper-case env vars: `wa_access_token` -> `WA_ACCESS_TOKEN`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "wabot_python"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    admin_api_key: str = ""       # empty disables auth on /api/*
    worker_count: int = 4         # background workers draining the queue

    # -------------------------------------------------- WhatsApp Cloud API
    wa_verify_token: str = "changeme"
    wa_access_token: str = ""
    wa_phone_number_id: str = ""
    wa_business_account_id: str = ""
    wa_app_secret: str = ""       # empty disables webhook signature checking
    wa_graph_version: str = "v26.0"
    wa_api_base: str = "https://graph.facebook.com"
    wa_allowed_numbers: str = ""  # empty means "reply to everyone"
    wa_mark_as_read: bool = True
    wa_typing_indicator: bool = True

    # -------------------------------------------------------------- routing
    default_provider: ProviderName = "openai"
    provider_fallback_order: str = "openai,anthropic,gemini"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_output_tokens: int = 1024
    temperature: float = 0.6
    request_timeout: float = 90.0

    # --------------------------------------------------------------- openai
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-terra"
    openai_transcribe_model: str = "gpt-transcribe"
    openai_base_url: str | None = None  # any OpenAI-compatible endpoint

    # ------------------------------------------------------------ anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    # --------------------------------------------------------------- gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_transcribe_model: str = "gemini-3.5-flash"

    # ---------------------------------------------------------------- media
    transcription_provider: Literal["openai", "gemini", "none"] = "openai"
    media_max_bytes: int = 15 * 1024 * 1024

    # --------------------------------------------------------------- memory
    history_max_messages: int = 20      # turns kept verbatim in the prompt
    summarize_after_messages: int = 40  # compress older turns past this mark

    # ------------------------------------------------------------- database
    database_url: str = "sqlite+aiosqlite:///./data/wabot.db"

    # ------------------------------------------------------------ validators
    @field_validator("wa_graph_version")
    @classmethod
    def _normalize_version(cls, value: str) -> str:
        """Accept both "26.0" and "v26.0"."""
        value = value.strip()
        return value if value.startswith("v") else f"v{value}"

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0")
        return value

    # ---------------------------------------------------------- derived data
    @property
    def graph_url(self) -> str:
        """e.g. https://graph.facebook.com/v26.0"""
        return f"{self.wa_api_base.rstrip('/')}/{self.wa_graph_version}"

    @property
    def messages_url(self) -> str:
        """Endpoint used to send messages and read receipts."""
        return f"{self.graph_url}/{self.wa_phone_number_id}/messages"

    @property
    def allowed_numbers(self) -> set[str]:
        """Allow-list parsed from the comma separated env var."""
        return {n.strip() for n in self.wa_allowed_numbers.split(",") if n.strip()}

    @property
    def fallback_order(self) -> list[str]:
        """Fallback chain, de-duplicated and normalized to lowercase."""
        seen: list[str] = []
        for name in self.provider_fallback_order.split(","):
            name = name.strip().lower()
            if name and name not in seen:
                seen.append(name)
        return seen

    def api_key_for(self, provider: str) -> str:
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
        }.get(provider, "")

    def model_for(self, provider: str) -> str:
        return {
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
            "gemini": self.gemini_model,
        }.get(provider, "")


@lru_cache
def get_settings() -> Settings:
    """Single cached settings instance shared by the whole app."""
    return Settings()
