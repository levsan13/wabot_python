"""MIME helpers — WhatsApp sends things like 'audio/ogg; codecs=opus'."""

from __future__ import annotations

import mimetypes

# mimetypes' database is thin on audio, so pin the ones WhatsApp actually sends.
_EXTRA = {
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/amr": ".amr",
    "audio/aac": ".aac",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}

# Image formats accepted by all three providers.
IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def clean(mime: str | None) -> str:
    """'audio/ogg; codecs=opus' -> 'audio/ogg'."""
    if not mime:
        return "application/octet-stream"
    return mime.split(";")[0].strip().lower()


def guess_extension(mime: str | None, default: str = "") -> str:
    """File extension for a MIME type — transcription APIs sniff the filename."""
    base = clean(mime)
    return _EXTRA.get(base) or mimetypes.guess_extension(base) or default


def is_image(mime: str | None) -> bool:
    return clean(mime) in IMAGE_MIMES


def is_pdf(mime: str | None) -> bool:
    return clean(mime) == "application/pdf"
