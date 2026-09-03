"""Turns WhatsApp media into something the models can actually read.

User-facing strings stay in Portuguese: `note` is sent straight to WhatsApp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Settings
from app.llm.base import Attachment, ProviderError
from app.llm.mime import clean, is_image, is_pdf
from app.llm.registry import ProviderRegistry
from app.whatsapp.client import WhatsAppClient, WhatsAppError
from app.whatsapp.schemas import IncomingMessage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedInput:
    """What will actually be sent to the model."""

    text: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    note: str | None = None       # warning to relay to the user
    media_kind: str | None = None
    media_mime: str | None = None
    transcript: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.attachments


class MediaService:
    """Downloads the media and decides whether it becomes text or an attachment."""

    def __init__(
        self, whatsapp: WhatsAppClient, registry: ProviderRegistry, settings: Settings
    ) -> None:
        self.whatsapp = whatsapp
        self.registry = registry
        self.settings = settings

    async def prepare(self, incoming: IncomingMessage) -> PreparedInput:
        """Resolve one inbound message into prompt-ready text and attachments."""
        kind = incoming.kind

        # ------------------------------------------------ no media involved
        if kind in ("text", "interactive", "button"):
            return PreparedInput(text=incoming.text)

        if kind == "reaction":
            return PreparedInput(
                text=f"(o usuário reagiu com {incoming.text} a uma mensagem anterior)"
            )

        if kind == "location":
            location = incoming.raw.get("location", {})
            return PreparedInput(
                text=(
                    "(localização enviada) "
                    f"latitude {location.get('latitude')}, longitude {location.get('longitude')}"
                    f" {location.get('name') or ''} {location.get('address') or ''}"
                ).strip()
            )

        if kind in ("video",):
            # None of the three providers takes video inline here.
            return PreparedInput(
                note="Ainda não consigo assistir vídeos. Me manda como texto, áudio, foto ou PDF?"
            )

        if not incoming.media_id:
            return PreparedInput(note="Não consegui identificar o arquivo enviado.")

        # ------------------------------------------------------- download it
        try:
            data, mime, filename = await self.whatsapp.download_media(incoming.media_id)
        except WhatsAppError as exc:
            logger.warning("Failed to download media %s: %s", incoming.media_id, exc)
            return PreparedInput(note=f"Não consegui baixar o arquivo. ({exc})")

        mime = clean(mime or incoming.mime_type)

        # -------------------------------------------------------------- audio
        if kind == "audio":
            # Voice notes become text, so any provider can answer them.
            audio = Attachment(kind="audio", mime_type=mime, data=data, filename=filename)
            try:
                transcript = await self.registry.transcribe(audio)
            except ProviderError as exc:
                logger.warning("Transcription failed: %s", exc)
                return PreparedInput(
                    note="Recebi seu áudio, mas não consegui transcrever agora. "
                    "Consegue mandar por escrito?",
                    media_kind="audio",
                    media_mime=mime,
                )
            if not transcript:
                return PreparedInput(
                    note="Seu áudio chegou vazio ou sem fala reconhecível.",
                    media_kind="audio",
                    media_mime=mime,
                )
            return PreparedInput(
                text=transcript,
                media_kind="audio",
                media_mime=mime,
                transcript=transcript,
            )

        # ----------------------------------------------------- image/sticker
        if kind in ("image", "sticker"):
            if not is_image(mime):
                return PreparedInput(
                    note=f"Formato de imagem não suportado ({mime}).",
                    media_kind=kind,
                    media_mime=mime,
                )
            # An image with no caption still needs an instruction.
            caption = incoming.text.strip() or "Descreva e analise a imagem enviada."
            return PreparedInput(
                text=caption,
                attachments=[
                    Attachment(kind="image", mime_type=mime, data=data, filename=filename)
                ],
                media_kind=kind,
                media_mime=mime,
            )

        # ---------------------------------------------------------- document
        if kind == "document":
            if is_pdf(mime):
                caption = incoming.text.strip() or (
                    f"Analise o documento anexo ({filename}) e resuma os pontos principais."
                )
                return PreparedInput(
                    text=caption,
                    attachments=[
                        Attachment(kind="document", mime_type=mime, data=data, filename=filename)
                    ],
                    media_kind="document",
                    media_mime=mime,
                )

            # Plain text files are cheaper inlined than sent as an attachment.
            if mime.startswith("text/") or mime in ("application/json", "application/xml"):
                try:
                    content = data.decode("utf-8", errors="replace")[:20000]
                except Exception:  # pragma: no cover - errors="replace" never raises
                    content = ""
                caption = incoming.text.strip() or "Analise o arquivo abaixo."
                return PreparedInput(
                    text=f"{caption}\n\n--- {filename} ---\n{content}",
                    media_kind="document",
                    media_mime=mime,
                )

            return PreparedInput(
                note=f"Consigo ler PDF e arquivos de texto; esse veio como {mime}.",
                media_kind="document",
                media_mime=mime,
            )

        return PreparedInput(note=f"Tipo de mensagem não suportado: {kind}.")
