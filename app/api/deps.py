"""Shared FastAPI dependencies and the application object graph."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

from app.config import Settings
from app.llm.registry import ProviderRegistry
from app.services.commands import CommandService
from app.services.conversation import ConversationService
from app.services.dispatcher import Dispatcher
from app.services.handler import MessageHandler
from app.services.media import MediaService
from app.whatsapp.client import WhatsAppClient


@dataclass
class AppContext:
    """Everything wired once at startup and shared by the routes.

    Dependencies are injected rather than imported, which is what lets the test
    suite swap the WhatsApp client and the provider registry for fakes.
    """

    settings: Settings
    whatsapp: WhatsAppClient
    registry: ProviderRegistry
    conversations: ConversationService
    media: MediaService
    commands: CommandService
    handler: MessageHandler
    dispatcher: Dispatcher

    async def aclose(self) -> None:
        """Drain the queue and close every HTTP client."""
        await self.dispatcher.stop()
        await self.whatsapp.aclose()
        await self.registry.aclose()


def build_context(settings: Settings) -> AppContext:
    """Compose the services, from the outermost dependency inwards."""
    whatsapp = WhatsAppClient(settings)
    registry = ProviderRegistry(settings)
    conversations = ConversationService(registry, settings)
    media = MediaService(whatsapp, registry, settings)
    commands = CommandService(registry, settings)
    handler = MessageHandler(settings, whatsapp, registry, conversations, media, commands)
    dispatcher = Dispatcher(handler, workers=settings.worker_count)
    return AppContext(
        settings=settings,
        whatsapp=whatsapp,
        registry=registry,
        conversations=conversations,
        media=media,
        commands=commands,
        handler=handler,
        dispatcher=dispatcher,
    )


def get_context(request: Request) -> AppContext:
    context = getattr(request.app.state, "context", None)
    if context is None:  # pragma: no cover - only if the lifespan failed
        raise HTTPException(status_code=503, detail="Application still starting up")
    return context


async def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> None:
    """Guard the admin routes. With no ADMIN_API_KEY set, everything is open."""
    expected = get_context(request).settings.admin_api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key"
        )
