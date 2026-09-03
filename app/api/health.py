"""Health and diagnostics route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import AppContext, get_context

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(context: AppContext = Depends(get_context)) -> dict:
    """Answers the questions asked most often when a bot goes quiet.

    Which providers hold a key, is the signature check on, is anything stuck in
    the queue.
    """
    settings = context.settings
    return {
        "status": "ok",
        "app": settings.app_name,
        "graph_version": settings.wa_graph_version,
        "whatsapp_configured": bool(settings.wa_access_token and settings.wa_phone_number_id),
        "signature_check": bool(settings.wa_app_secret),
        "providers": {
            "default": settings.default_provider,
            "available": context.registry.available(),
            "fallback_order": settings.fallback_order,
        },
        "transcription": {
            "provider": settings.transcription_provider,
            "ready": context.registry.transcriber() is not None,
        },
        "queue_pending": context.dispatcher.pending,
    }
