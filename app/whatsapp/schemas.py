"""Webhook payload models plus normalization into an internal shape.

Meta's JSON is deeply nested (entry > changes > value > messages) and grows new
fields regularly, so the models are permissive and everything the app needs is
flattened into `IncomingMessage`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

MEDIA_TYPES = ("image", "audio", "video", "document", "sticker")


class _Lenient(BaseModel):
    """Meta adds fields often — never blow up because of one."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class WAText(_Lenient):
    # `body` may come back null on edited or unsupported messages.
    body: str | None = ""


class WAMedia(_Lenient):
    """Shared shape of image / audio / video / document / sticker payloads."""

    id: str | None = None
    mime_type: str | None = None
    sha256: str | None = None
    caption: str | None = None
    filename: str | None = None
    voice: bool | None = None  # True for voice notes, absent for audio files


class WAContext(_Lenient):
    """Present when the user replied to a specific message."""

    id: str | None = None
    from_: str | None = Field(default=None, alias="from")
    forwarded: bool | None = None


class WAMessage(_Lenient):
    id: str
    from_: str = Field(alias="from")  # "from" is a Python keyword
    timestamp: str | None = None
    type: str = "unknown"

    text: WAText | None = None
    image: WAMedia | None = None
    audio: WAMedia | None = None
    video: WAMedia | None = None
    document: WAMedia | None = None
    sticker: WAMedia | None = None

    interactive: dict[str, Any] | None = None
    button: dict[str, Any] | None = None
    reaction: dict[str, Any] | None = None
    location: dict[str, Any] | None = None
    contacts: list[dict[str, Any]] | None = None
    errors: list[dict[str, Any]] | None = None
    context: WAContext | None = None


class WAProfile(_Lenient):
    name: str | None = None


class WAContact(_Lenient):
    wa_id: str | None = None
    profile: WAProfile | None = None


class WAMetadata(_Lenient):
    display_phone_number: str | None = None
    phone_number_id: str | None = None


class WAStatus(_Lenient):
    """Delivery receipts: sent / delivered / read / failed."""

    id: str | None = None
    status: str | None = None
    recipient_id: str | None = None
    timestamp: str | None = None
    errors: list[dict[str, Any]] | None = None


class WAValue(_Lenient):
    messaging_product: str | None = None
    metadata: WAMetadata | None = None
    contacts: list[WAContact] = Field(default_factory=list)
    messages: list[WAMessage] = Field(default_factory=list)
    statuses: list[WAStatus] = Field(default_factory=list)


class WAChange(_Lenient):
    field: str | None = None  # "messages" for everything this bot cares about
    value: WAValue = Field(default_factory=WAValue)


class WAEntry(_Lenient):
    id: str | None = None
    changes: list[WAChange] = Field(default_factory=list)


class WebhookEnvelope(_Lenient):
    object: str | None = None
    entry: list[WAEntry] = Field(default_factory=list)


# ------------------------------------------------------------------ normalized
@dataclass(slots=True)
class IncomingMessage:
    """An inbound message flattened to what the application actually uses."""

    message_id: str
    from_number: str
    phone_number_id: str | None = None
    contact_name: str | None = None
    sent_at: datetime | None = None
    kind: str = "unknown"          # text | image | audio | document | sticker | ...
    text: str = ""                 # body, caption, or button label
    media_id: str | None = None
    mime_type: str | None = None
    filename: str | None = None
    reply_to: str | None = None    # wamid this message replies to
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_media(self) -> bool:
        return self.media_id is not None

    @property
    def is_command(self) -> bool:
        return self.text.strip().startswith("/")


def _as_datetime(timestamp: str | None) -> datetime | None:
    """Meta sends a Unix timestamp as a string."""
    if not timestamp:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def _interactive_text(message: WAMessage) -> str:
    """Extract the label a user tapped on a button or a list."""
    data = message.interactive or {}
    for key in ("button_reply", "list_reply"):
        reply = data.get(key) or {}
        title = reply.get("title") or reply.get("id")
        if title:
            return str(title)
    if message.button:
        return str(message.button.get("text") or message.button.get("payload") or "")
    return ""


def normalize_message(
    message: WAMessage, metadata: WAMetadata | None, contacts: list[WAContact]
) -> IncomingMessage:
    """Flatten one WAMessage, pulling the sender's profile name from `contacts`."""
    name = None
    for contact in contacts:
        if contact.wa_id == message.from_ and contact.profile:
            name = contact.profile.name
            break
    if name is None and contacts and contacts[0].profile:
        name = contacts[0].profile.name

    incoming = IncomingMessage(
        message_id=message.id,
        from_number=message.from_,
        phone_number_id=metadata.phone_number_id if metadata else None,
        contact_name=name,
        sent_at=_as_datetime(message.timestamp),
        kind=message.type,
        reply_to=message.context.id if message.context else None,
        raw=message.model_dump(by_alias=True, exclude_none=True),
    )

    if message.type == "text" and message.text:
        incoming.text = message.text.body or ""
    elif message.type in ("interactive", "button"):
        incoming.text = _interactive_text(message)
    elif message.type in MEDIA_TYPES:
        media: WAMedia | None = getattr(message, message.type, None)
        if media:
            incoming.media_id = media.id
            incoming.mime_type = media.mime_type
            incoming.filename = media.filename
            incoming.text = media.caption or ""
    elif message.type == "reaction" and message.reaction:
        incoming.text = str(message.reaction.get("emoji") or "")

    return incoming


@dataclass(slots=True)
class WebhookParse:
    """Result of parsing one webhook body."""

    messages: list[IncomingMessage] = field(default_factory=list)
    statuses: list[WAStatus] = field(default_factory=list)


def parse_webhook(payload: dict[str, Any]) -> WebhookParse:
    """Turn Meta's raw JSON into normalized messages and delivery statuses.

    A single POST can legitimately carry several entries and changes, so
    everything is flattened into two lists.
    """
    envelope = WebhookEnvelope.model_validate(payload)
    parsed = WebhookParse()

    for entry in envelope.entry:
        for change in entry.changes:
            value = change.value
            for message in value.messages:
                parsed.messages.append(normalize_message(message, value.metadata, value.contacts))
            parsed.statuses.extend(value.statuses)

    return parsed
