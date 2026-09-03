"""The parser copes with the payload shapes Meta actually sends."""

from __future__ import annotations

from app.whatsapp.schemas import parse_webhook
from tests.conftest import webhook_payload


def test_plain_text():
    parsed = parse_webhook(webhook_payload("bom dia"))
    assert len(parsed.messages) == 1
    message = parsed.messages[0]
    assert message.text == "bom dia"
    assert message.from_number == "5585999990000"
    assert message.contact_name == "Levy"  # pulled from the contacts block
    assert message.kind == "text"
    assert not message.has_media


def test_image_with_caption():
    parsed = parse_webhook(
        webhook_payload(
            message_id="wamid.img",
            type="image",
            text=None,
            image={"id": "MEDIA_1", "mime_type": "image/jpeg", "caption": "o que é isso?"},
        )
    )
    message = parsed.messages[0]
    assert message.kind == "image"
    assert message.media_id == "MEDIA_1"
    assert message.mime_type == "image/jpeg"
    # The caption becomes the prompt text.
    assert message.text == "o que é isso?"
    assert message.has_media


def test_voice_note():
    parsed = parse_webhook(
        webhook_payload(
            message_id="wamid.aud",
            type="audio",
            text=None,
            audio={"id": "MEDIA_2", "mime_type": "audio/ogg; codecs=opus", "voice": True},
        )
    )
    assert parsed.messages[0].kind == "audio"
    assert parsed.messages[0].media_id == "MEDIA_2"


def test_pdf_document():
    parsed = parse_webhook(
        webhook_payload(
            message_id="wamid.doc",
            type="document",
            text=None,
            document={
                "id": "MEDIA_3",
                "mime_type": "application/pdf",
                "filename": "contrato.pdf",
            },
        )
    )
    message = parsed.messages[0]
    assert message.filename == "contrato.pdf"
    assert message.kind == "document"


def test_interactive_button_becomes_text():
    parsed = parse_webhook(
        webhook_payload(
            message_id="wamid.int",
            type="interactive",
            text=None,
            interactive={
                "type": "button_reply",
                "button_reply": {"id": "sim", "title": "Quero sim"},
            },
        )
    )
    assert parsed.messages[0].text == "Quero sim"


def test_status_is_not_a_message():
    """Delivery receipts must not be answered."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "statuses": [
                                {
                                    "id": "wamid.x",
                                    "status": "delivered",
                                    "recipient_id": "5585999990000",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    parsed = parse_webhook(payload)
    assert parsed.messages == []
    assert parsed.statuses[0].status == "delivered"


def test_unknown_fields_do_not_break_parsing():
    """Meta adds fields without warning; the models are permissive on purpose."""
    payload = webhook_payload("oi")
    payload["entry"][0]["changes"][0]["value"]["campo_do_futuro"] = {"a": 1}
    payload["entry"][0]["changes"][0]["value"]["messages"][0]["novidade"] = True
    assert parse_webhook(payload).messages[0].text == "oi"


def test_command_is_detected():
    parsed = parse_webhook(webhook_payload("/status"))
    assert parsed.messages[0].is_command
