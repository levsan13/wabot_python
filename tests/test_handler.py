"""End-to-end flow: webhook -> handler -> reply, with fake providers.

Assertions on Portuguese strings are intentional: those are what the bot says.
"""

from __future__ import annotations

import pytest

from app.db import repo
from app.db.base import session_scope
from app.llm.base import Attachment
from app.whatsapp.schemas import parse_webhook
from tests.conftest import webhook_payload


def first_message(**kwargs):
    """Shortcut: build a payload and take the single message out of it."""
    return parse_webhook(webhook_payload(**kwargs)).messages[0]


async def test_text_produces_an_answer(context, whatsapp, registry):
    await context.handler.handle(first_message(text="qual a capital do Ceará?"))

    assert whatsapp.last_message == "resposta de teste"
    assert whatsapp.read == ["wamid.1"]  # marked as read
    async with session_scope() as session:
        conversation = await repo.get_or_create_conversation(session, "5585999990000")
        assert conversation.message_count == 2  # question + answer
        assert conversation.display_name == "Levy"


async def test_duplicate_event_is_ignored(context, whatsapp):
    """Meta redelivers when the endpoint is slow; the bot must not answer twice."""
    message = first_message(text="oi")
    await context.handler.handle(message)
    await context.handler.handle(message)
    assert len(whatsapp.sent) == 1


async def test_history_reaches_the_model(context, registry):
    await context.handler.handle(first_message(text="me chamo Levy", message_id="wamid.a"))
    await context.handler.handle(first_message(text="como me chamo?", message_id="wamid.b"))

    turns = registry.calls[-1]["turns"]
    assert [t.role for t in turns] == ["user", "assistant", "user"]
    assert turns[0].text == "me chamo Levy"


async def test_command_does_not_call_the_llm(context, whatsapp, registry):
    await context.handler.handle(first_message(text="/ajuda"))
    assert registry.calls == []  # no tokens spent
    assert "Comandos disponíveis" in whatsapp.last_message


async def test_provider_switch_persists(context, whatsapp, registry):
    await context.handler.handle(first_message(text="/provedor gemini", message_id="wamid.p"))
    assert "gemini" in whatsapp.last_message

    # The next ordinary message must go to the newly chosen provider.
    await context.handler.handle(first_message(text="e aí?", message_id="wamid.q"))
    assert registry.calls[-1]["provider"] == "gemini"


async def test_reset_clears_history(context, whatsapp):
    await context.handler.handle(first_message(text="oi", message_id="wamid.1"))
    await context.handler.handle(first_message(text="/reset", message_id="wamid.2"))
    assert "Histórico apagado" in whatsapp.last_message

    async with session_scope() as session:
        conversation = await repo.get_or_create_conversation(session, "5585999990000")
        assert await repo.count_messages(session, conversation.id) == 0


async def test_audio_is_transcribed(context, whatsapp, registry):
    """A voice note reaches the model as text, not as an attachment."""
    whatsapp.media["MEDIA_AUD"] = (b"\x00\x01ogg", "audio/ogg", "audio.ogg")
    message = first_message(
        message_id="wamid.aud",
        type="audio",
        text=None,
        audio={"id": "MEDIA_AUD", "mime_type": "audio/ogg; codecs=opus"},
    )
    await context.handler.handle(message)

    turns = registry.calls[-1]["turns"]
    assert turns[-1].text == "transcrição de teste"
    assert turns[-1].attachments == []


async def test_image_becomes_an_attachment(context, whatsapp, registry):
    whatsapp.media["MEDIA_IMG"] = (b"\xff\xd8\xff", "image/jpeg", "foto.jpg")
    message = first_message(
        message_id="wamid.img",
        type="image",
        text=None,
        image={"id": "MEDIA_IMG", "mime_type": "image/jpeg", "caption": "o que tem aqui?"},
    )
    await context.handler.handle(message)

    attachments: list[Attachment] = registry.calls[-1]["turns"][-1].attachments
    assert len(attachments) == 1
    assert attachments[0].kind == "image"
    assert attachments[0].mime_type == "image/jpeg"
    assert registry.calls[-1]["turns"][-1].text == "o que tem aqui?"


async def test_video_is_politely_refused(context, whatsapp, registry):
    message = first_message(
        message_id="wamid.vid",
        type="video",
        text=None,
        video={"id": "MEDIA_VID", "mime_type": "video/mp4"},
    )
    await context.handler.handle(message)
    assert "vídeos" in whatsapp.last_message
    assert registry.calls == []  # never reaches a provider


async def test_provider_failure_is_reported_to_the_user(context, whatsapp, registry):
    registry.fail = True
    await context.handler.handle(first_message(text="oi"))
    assert "problema para gerar a resposta" in whatsapp.last_message


async def test_allow_list_blocks_unknown_numbers(context, whatsapp):
    context.settings.wa_allowed_numbers = "5511999999999"
    await context.handler.handle(first_message(text="oi"))
    assert whatsapp.sent == []


@pytest.mark.parametrize("command", ["/status", "/modelos", "/persona seja formal", "/modelo x"])
async def test_every_command_answers_something(context, whatsapp, command):
    """Smoke test so no command silently returns nothing."""
    await context.handler.handle(first_message(text=command, message_id=f"wamid{command}"))
    assert whatsapp.last_message
