"""Shared fixtures: temporary database, fake providers and fake WhatsApp.

Nothing here touches the network — the fakes stand in for the SDK clients so the
whole pipeline can be exercised offline.
"""

from __future__ import annotations

import pytest

from app.api.deps import AppContext
from app.config import Settings
from app.db.base import create_all, dispose_engine, init_engine
from app.llm.base import Attachment, ChatTurn, LLMReply
from app.llm.registry import ProviderRegistry
from app.services.commands import CommandService
from app.services.conversation import ConversationService
from app.services.dispatcher import Dispatcher
from app.services.handler import MessageHandler
from app.services.media import MediaService


class FakeRegistry(ProviderRegistry):
    """Real registry (so resolve/available behave) with generation stubbed out.

    Subclassing rather than mocking keeps provider resolution, the fallback
    chain and the availability checks under test.
    """

    def __init__(self, settings: Settings, reply_text: str = "resposta de teste") -> None:
        super().__init__(settings)
        self.reply_text = reply_text
        self.calls: list[dict] = []  # inspected by the tests
        self.fail = False            # flip to simulate a provider outage

    async def chat(self, turns: list[ChatTurn], **kwargs) -> LLMReply:
        self.calls.append({"turns": turns, **kwargs})
        if self.fail:
            from app.llm.base import ProviderError

            raise ProviderError("simulated failure")
        return LLMReply(
            text=self.reply_text,
            provider=self.resolve(kwargs.get("provider")),
            model=kwargs.get("model") or "modelo-de-teste",
            input_tokens=10,
            output_tokens=5,
            latency_ms=42,
        )

    async def transcribe(self, audio: Attachment, language: str | None = "pt") -> str:
        return "transcrição de teste"

    def transcriber(self):
        # Always "available" in tests, no API key needed.
        return self.get("openai")

    async def aclose(self) -> None:
        return None


class FakeWhatsApp:
    """Stands in for the HTTP client: records what would have been sent."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.read: list[str] = []
        # media_id -> (bytes, mime, filename); tests fill this in.
        self.media: dict[str, tuple[bytes, str, str]] = {}

    async def send_text(self, to: str, body: str, **_kwargs) -> list[str]:
        self.sent.append((to, body))
        return [f"wamid.fake{len(self.sent)}"]

    async def mark_read(self, message_id: str, **_kwargs) -> None:
        self.read.append(message_id)

    async def send_reaction(self, *_args, **_kwargs) -> None:
        return None

    async def download_media(self, media_id: str, *_args, **_kwargs):
        if media_id not in self.media:
            from app.whatsapp.client import WhatsAppError

            raise WhatsAppError(f"media {media_id} does not exist")
        return self.media[media_id]

    async def aclose(self) -> None:
        return None

    @property
    def last_message(self) -> str:
        return self.sent[-1][1] if self.sent else ""


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Fully configured settings pointing at a throwaway SQLite file.

    `_env_file=None` keeps a developer's real .env out of the test run.
    """
    return Settings(
        wa_verify_token="token-de-teste",
        wa_access_token="EAAtest",
        wa_phone_number_id="123456",
        wa_app_secret="segredo",
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test",
        gemini_api_key="gm-test",
        admin_api_key="",
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        summarize_after_messages=0,  # summarization off unless a test wants it
        _env_file=None,
    )


@pytest.fixture
async def database(settings):
    """Fresh schema per test; the engine is a module singleton, so dispose it."""
    init_engine(settings.database_url)
    await create_all()
    yield
    await dispose_engine()


@pytest.fixture
def registry(settings) -> FakeRegistry:
    return FakeRegistry(settings)


@pytest.fixture
def whatsapp() -> FakeWhatsApp:
    return FakeWhatsApp()


@pytest.fixture
def context(settings, registry, whatsapp, database) -> AppContext:
    """The real object graph, with the two outbound edges faked."""
    conversations = ConversationService(registry, settings)
    media = MediaService(whatsapp, registry, settings)
    commands = CommandService(registry, settings)
    handler = MessageHandler(settings, whatsapp, registry, conversations, media, commands)
    return AppContext(
        settings=settings,
        whatsapp=whatsapp,
        registry=registry,
        conversations=conversations,
        media=media,
        commands=commands,
        handler=handler,
        dispatcher=Dispatcher(handler, workers=1),
    )


def webhook_payload(
    text: str | None = "oi", message_id: str = "wamid.1", **overrides
) -> dict:
    """Build a payload shaped exactly like Meta's.

    `text=None` drops the text block (for media messages); `overrides` replaces
    keys on the message object, e.g. `type="image", image={...}`.
    """
    message = {
        "from": "5585999990000",
        "id": message_id,
        "timestamp": "1750000000",
        "type": "text",
        "text": {"body": text},
    }
    if text is None:
        message.pop("text")
    message.update(overrides)
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "5585111111111",
                                "phone_number_id": "123456",
                            },
                            "contacts": [
                                {"profile": {"name": "Levy"}, "wa_id": "5585999990000"}
                            ],
                            "messages": [message],
                        },
                    }
                ],
            }
        ],
    }
