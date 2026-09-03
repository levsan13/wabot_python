"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import admin, health, webhook
from app.api.deps import build_context
from app.config import get_settings
from app.db.base import create_all, dispose_engine, init_engine
from app.logging_conf import setup_logging

logger = logging.getLogger(__name__)

DESCRIPTION = """
WhatsApp bot (Meta Cloud API) with three pluggable LLM providers:
**OpenAI**, **Anthropic** and **Google Gemini**.

* `POST /webhook` — receives Meta events (signature checked, work queued in the background)
* `POST /api/chat` — talk to the LLM without going through WhatsApp
* `POST /api/messages` — send a message from the bot's number
"""


def _startup_report(context) -> None:
    """Print the effective configuration and warn about the usual foot-guns."""
    settings = context.settings
    available = context.registry.available()

    logger.info("=" * 62)
    logger.info("%s ready", settings.app_name)
    logger.info("Graph API .......... %s", settings.wa_graph_version)
    logger.info("Providers .......... %s", ", ".join(available) or "NONE")
    logger.info("Default ............ %s", settings.default_provider)
    logger.info("Transcription ...... %s", getattr(context.registry.transcriber(), "name", "off"))
    logger.info("=" * 62)

    if not available:
        logger.warning(
            "No LLM API key configured — the bot will receive messages and fail to "
            "answer. Set OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY."
        )
    if settings.default_provider not in available and available:
        logger.warning(
            "DEFAULT_PROVIDER=%s has no API key; falling back to '%s'.",
            settings.default_provider,
            available[0],
        )
    if not settings.wa_access_token or not settings.wa_phone_number_id:
        logger.warning("WA_ACCESS_TOKEN/WA_PHONE_NUMBER_ID are empty — cannot send anything.")
    if not settings.wa_app_secret:
        logger.warning(
            "WA_APP_SECRET is empty: webhook signatures will NOT be validated. "
            "Do not ship this to production."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the database, providers and worker queue; tear them down cleanly."""
    settings = get_settings()
    setup_logging(settings.log_level)

    init_engine(settings.database_url)
    await create_all()

    context = build_context(settings)
    await context.dispatcher.start()
    app.state.context = context
    _startup_report(context)

    try:
        yield
    finally:
        await context.aclose()
        await dispose_engine()
        logger.info("Shut down.")


def create_app() -> FastAPI:
    """Application factory — the tests build their own instance with fakes."""
    settings = get_settings()
    app = FastAPI(
        title="wabot_python",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(webhook.router)
    app.include_router(admin.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "app": settings.app_name,
            "docs": "/docs",
            "webhook": "/webhook",
            "health": "/health",
        }

    return app


app = create_app()
