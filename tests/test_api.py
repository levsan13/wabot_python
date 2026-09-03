"""HTTP routes: webhook verification, event intake and the REST API."""

from __future__ import annotations

import json

import httpx
import pytest

from app.main import create_app
from app.whatsapp.security import compute_signature
from tests.conftest import webhook_payload


@pytest.fixture
def client(context):
    """App wired to the fake context.

    ASGITransport does not run the lifespan, which is exactly what we want:
    `app.state.context` is set by hand so nothing real is started.
    """
    app = create_app()
    app.state.context = context
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_webhook_verification(client, context):
    async with client:
        response = await client.get(
            "/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": context.settings.wa_verify_token,
                "hub.challenge": "1234567890",
            },
        )
    assert response.status_code == 200
    # Meta expects the challenge echoed back verbatim, as plain text.
    assert response.text == "1234567890"


async def test_webhook_verification_with_wrong_token(client):
    async with client:
        response = await client.get(
            "/webhook",
            params={"hub.mode": "subscribe", "hub.verify_token": "errado", "hub.challenge": "1234"},
        )
    assert response.status_code == 403


async def test_webhook_queues_the_message(client, context):
    payload = webhook_payload("olá")
    raw = json.dumps(payload).encode()
    signature = compute_signature(context.settings.wa_app_secret, raw)

    async with client:
        response = await client.post(
            "/webhook", content=raw, headers={"X-Hub-Signature-256": signature}
        )

    # Answers immediately and hands the work to the dispatcher.
    assert response.status_code == 200
    assert response.json()["queued"] == 1
    assert context.dispatcher.pending == 1


async def test_webhook_refuses_a_bad_signature(client, context):
    raw = json.dumps(webhook_payload("olá")).encode()
    async with client:
        response = await client.post(
            "/webhook", content=raw, headers={"X-Hub-Signature-256": "sha256=errado"}
        )
    assert response.status_code == 403
    assert context.dispatcher.pending == 0


async def test_broken_json_still_returns_200(client, context):
    """Answering 4xx would only make Meta retry a body that cannot be parsed."""
    raw = b"{nao eh json"
    signature = compute_signature(context.settings.wa_app_secret, raw)
    async with client:
        response = await client.post(
            "/webhook", content=raw, headers={"X-Hub-Signature-256": signature}
        )
    assert response.status_code == 200


async def test_health(client):
    async with client:
        response = await client.get("/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert set(body["providers"]["available"]) == {"openai", "anthropic", "gemini"}


async def test_api_chat(client):
    async with client:
        response = await client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "oi"}]}
        )
    assert response.status_code == 200
    assert response.json()["text"] == "resposta de teste"


async def test_api_send_message(client, whatsapp):
    async with client:
        response = await client.post("/api/messages", json={"to": "5585999990000", "text": "oi"})
    assert response.status_code == 200
    assert whatsapp.sent[-1] == ("5585999990000", "oi")


async def test_api_requires_the_key_when_configured(client, context):
    context.settings.admin_api_key = "secreta"
    async with client:
        denied = await client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "oi"}]}
        )
        allowed = await client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "oi"}]},
            headers={"X-API-Key": "secreta"},
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200


async def test_api_lists_conversations(client, context):
    from app.whatsapp.schemas import parse_webhook

    await context.handler.handle(parse_webhook(webhook_payload("oi")).messages[0])
    async with client:
        response = await client.get("/api/conversations")
    assert response.status_code == 200
    assert response.json()[0]["wa_id"] == "5585999990000"
