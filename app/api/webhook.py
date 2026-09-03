"""Meta webhook routes: verification (GET) and event intake (POST)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api.deps import AppContext, get_context
from app.whatsapp.schemas import parse_webhook
from app.whatsapp.security import SIGNATURE_HEADER, verify_signature

logger = logging.getLogger(__name__)
router = APIRouter(tags=["webhook"])


@router.get("/webhook", include_in_schema=False)
async def verify_webhook(
    context: AppContext = Depends(get_context),
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
) -> Response:
    """Handshake Meta performs when the webhook URL is registered.

    Echo back `hub.challenge` as plain text when the token matches.
    """
    if hub_mode == "subscribe" and hub_verify_token == context.settings.wa_verify_token:
        logger.info("Webhook successfully verified by Meta")
        return PlainTextResponse(content=hub_challenge or "", status_code=200)

    logger.warning("Webhook verification refused (token mismatch)")
    return PlainTextResponse(content="verification failed", status_code=403)


@router.post("/webhook")
async def receive_webhook(request: Request, context: AppContext = Depends(get_context)) -> Response:
    """Receive events, validate the signature and enqueue the work.

    Always answers 200 quickly: any delay makes Meta redeliver the event. Even
    a malformed body gets a 200, because retrying it would not help either.
    """
    # Raw bytes: re-serialized JSON would not match the HMAC.
    raw = await request.body()

    if not verify_signature(
        context.settings.wa_app_secret, raw, request.headers.get(SIGNATURE_HEADER)
    ):
        logger.warning("Invalid webhook signature — event discarded")
        return JSONResponse({"status": "invalid signature"}, status_code=status.HTTP_403_FORBIDDEN)

    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        logger.warning("Webhook body was not valid JSON")
        return JSONResponse({"status": "invalid json"}, status_code=200)

    try:
        parsed = parse_webhook(payload)
    except Exception:
        logger.exception("Could not interpret the webhook payload")
        return JSONResponse({"status": "unparsed"}, status_code=200)

    # Delivery receipts are only logged; they carry no work.
    for status_update in parsed.statuses:
        logger.debug("Status %s for message %s", status_update.status, status_update.id)

    queued = 0
    for message in parsed.messages:
        if await context.dispatcher.submit(message):
            queued += 1

    if queued:
        logger.info("%d message(s) queued", queued)
    return JSONResponse({"status": "ok", "queued": queued}, status_code=200)
