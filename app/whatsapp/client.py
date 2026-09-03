"""HTTP client for the WhatsApp Cloud API (sending, receipts, media)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings
from app.llm.mime import clean, guess_extension

logger = logging.getLogger(__name__)

TEXT_LIMIT = 4096   # hard limit of a message body in the Cloud API
SAFE_CHUNK = 3800   # margin so formatting never pushes a chunk over the limit


class WhatsAppError(RuntimeError):
    """Any non-2xx answer from the Graph API, or a network failure."""


def split_text(text: str, limit: int = SAFE_CHUNK) -> list[str]:
    """Split long answers, preferring paragraph > line > space boundaries.

    Models happily write past 4096 characters; sending that raw is a 400 from
    the API, so anything long is delivered as several messages.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        # No decent boundary in the second half? Hard-cut rather than emit a tiny chunk.
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


class WhatsAppClient:
    """Thin wrapper over the Graph API endpoints this bot needs."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        # The injected client is what the tests replace.
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=60.0),
            headers={
                "Authorization": f"Bearer {settings.wa_access_token}",
                "User-Agent": f"{settings.app_name}/1.0",
            },
        )

    # ------------------------------------------------------------- internal
    async def _post_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to /{phone_number_id}/messages, raising on any error."""
        try:
            response = await self._client.post(self.settings.messages_url, json=payload)
        except httpx.HTTPError as exc:
            raise WhatsAppError(f"Network error calling the Graph API: {exc}") from exc

        if response.status_code >= 400:
            raise WhatsAppError(
                f"Graph API returned {response.status_code}: {response.text[:500]}"
            )
        return response.json()

    # -------------------------------------------------------------- sending
    async def send_text(
        self, to: str, body: str, *, preview_url: bool = False, reply_to: str | None = None
    ) -> list[str]:
        """Send text, splitting it when needed. Returns the resulting wamids."""
        sent: list[str] = []
        chunks = split_text(body)
        for index, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {"preview_url": preview_url, "body": chunk},
            }
            # Only the first chunk quotes the original message.
            if reply_to and index == 0:
                payload["context"] = {"message_id": reply_to}
            data = await self._post_message(payload)
            sent.extend(m.get("id") for m in data.get("messages", []) if m.get("id"))
        return sent

    async def mark_read(self, message_id: str, *, typing: bool = False) -> None:
        """Mark as read and optionally show "typing…" for up to 25 seconds.

        Both live on the same endpoint, so one call does the pair. Failures are
        cosmetic and must never abort answering.
        """
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        if typing:
            payload["typing_indicator"] = {"type": "text"}
        try:
            await self._post_message(payload)
        except WhatsAppError as exc:
            logger.debug("Could not mark %s as read: %s", message_id, exc)

    async def send_reaction(self, to: str, message_id: str, emoji: str) -> None:
        """React to a message with an emoji (empty string removes the reaction)."""
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "reaction",
            "reaction": {"message_id": message_id, "emoji": emoji},
        }
        try:
            await self._post_message(payload)
        except WhatsAppError as exc:
            logger.debug("Could not react to %s: %s", message_id, exc)

    # ---------------------------------------------------------------- media
    async def media_metadata(self, media_id: str) -> dict[str, Any]:
        """Step 1 of a download: resolve the id into a short-lived URL."""
        url = f"{self.settings.graph_url}/{media_id}"
        response = await self._client.get(url)
        if response.status_code >= 400:
            raise WhatsAppError(
                f"Could not fetch metadata for media {media_id}: "
                f"{response.status_code} {response.text[:300]}"
            )
        return response.json()

    async def download_media(
        self, media_id: str, max_bytes: int | None = None
    ) -> tuple[bytes, str, str]:
        """Download a media file. Returns (bytes, mime_type, filename).

        Two hops: metadata first, then the URL it hands back — which still
        requires the bearer token.
        """
        max_bytes = max_bytes or self.settings.media_max_bytes
        meta = await self.media_metadata(media_id)
        url = meta.get("url")
        mime = clean(meta.get("mime_type"))
        size = int(meta.get("file_size") or 0)

        if not url:
            raise WhatsAppError(f"Media {media_id} has no download URL.")
        # Check the advertised size before spending bandwidth on it.
        if size and size > max_bytes:
            raise WhatsAppError(f"File too large ({size} bytes > limit of {max_bytes}).")

        response = await self._client.get(url, follow_redirects=True)
        if response.status_code >= 400:
            raise WhatsAppError(f"Download of media {media_id} failed: {response.status_code}")

        content = response.content
        # And check again, in case the metadata lied about the size.
        if len(content) > max_bytes:
            raise WhatsAppError(f"File too large ({len(content)} bytes > limit of {max_bytes}).")

        filename = meta.get("filename") or f"{media_id}{guess_extension(mime, '.bin')}"
        return content, mime, filename

    async def aclose(self) -> None:
        await self._client.aclose()
