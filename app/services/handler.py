"""Orchestrates what happens when a message arrives."""

from __future__ import annotations

import logging

from app.config import Settings
from app.db import repo
from app.db.base import session_scope
from app.db.models import Conversation
from app.llm.base import ChatTurn, LLMReply, ProviderError
from app.llm.registry import ProviderRegistry
from app.services.commands import CommandService
from app.services.conversation import ConversationService
from app.services.media import MediaService, PreparedInput
from app.whatsapp.client import WhatsAppClient, WhatsAppError
from app.whatsapp.schemas import IncomingMessage

logger = logging.getLogger(__name__)

# Shown to the user when every provider in the chain failed.
FALLBACK_ERROR = (
    "Tive um problema para gerar a resposta agora. "
    "Tenta de novo em instantes — se persistir, use /status para ver o provedor ativo."
)


class MessageHandler:
    """Ties together dedup, commands, media, memory, the LLM and the reply."""

    def __init__(
        self,
        settings: Settings,
        whatsapp: WhatsAppClient,
        registry: ProviderRegistry,
        conversations: ConversationService,
        media: MediaService,
        commands: CommandService,
    ) -> None:
        self.settings = settings
        self.whatsapp = whatsapp
        self.registry = registry
        self.conversations = conversations
        self.media = media
        self.commands = commands

    # ----------------------------------------------------------------- flow
    async def handle(self, incoming: IncomingMessage) -> None:
        """Full pipeline for one inbound message. Runs off the request path.

        Database transactions are deliberately short and never wrap the LLM
        call: a slow provider must not hold a SQLite write lock open.
        """
        allowed = self.settings.allowed_numbers
        if allowed and incoming.from_number not in allowed:
            logger.info("Number %s is not allow-listed — ignored", incoming.from_number)
            return

        # 1) Dedup + conversation lookup, in one short transaction.
        async with session_scope() as session:
            if not await repo.claim_event(session, incoming.message_id):
                logger.debug("Event %s already processed", incoming.message_id)
                return
            conversation = await repo.get_or_create_conversation(
                session, incoming.from_number, incoming.contact_name
            )
            conversation_id = conversation.id
            is_blocked = conversation.is_blocked

        if is_blocked:
            logger.info("Conversation %s is blocked", incoming.from_number)
            return

        if self.settings.wa_mark_as_read:
            await self.whatsapp.mark_read(
                incoming.message_id, typing=self.settings.wa_typing_indicator
            )

        # 2) Commands answer on their own, with no LLM call.
        text = incoming.text or ""
        if incoming.kind in ("text", "interactive", "button") and CommandService.is_command(text):
            async with session_scope() as session:
                conversation = await session.get(Conversation, conversation_id)
                outcome = await self.commands.execute(session, conversation, text)
            if outcome.reply:
                await self._reply(incoming, outcome.reply)
            return

        # 3) Media -> text and/or attachments.
        prepared = await self.media.prepare(incoming)
        if prepared.note and prepared.is_empty:
            # Nothing usable: relay the explanation and stop.
            await self._reply(incoming, prepared.note)
            return
        if prepared.is_empty:
            return

        # 4) Build the prompt from persona + summary + recent history.
        async with session_scope() as session:
            conversation = await session.get(Conversation, conversation_id)
            system = self.conversations.system_prompt(conversation)
            turns = await self.conversations.history_turns(session, conversation)
            provider = conversation.provider
            model = conversation.model

            await repo.add_message(
                session,
                conversation,
                role="user",
                content=prepared.text,
                media_kind=prepared.media_kind,
                media_mime=prepared.media_mime,
                wa_message_id=incoming.message_id,
            )

        # The current turn carries the attachments; history is text only.
        turns.append(ChatTurn(role="user", text=prepared.text, attachments=prepared.attachments))

        # 5) Call the model, outside of any open transaction.
        try:
            reply: LLMReply = await self.registry.chat(
                turns, system=system, provider=provider, model=model
            )
        except ProviderError as exc:
            logger.error("All providers failed for %s: %s", incoming.from_number, exc)
            await self._reply(incoming, FALLBACK_ERROR)
            return

        answer = reply.text.strip() or "Não consegui formular uma resposta para isso."
        if prepared.note:
            # e.g. an attachment was skipped but the text still got an answer.
            answer = f"{prepared.note}\n\n{answer}"

        # 6) Reply first, persist second — the user should never wait on the DB.
        sent_ids = await self._reply(incoming, answer)

        async with session_scope() as session:
            conversation = await session.get(Conversation, conversation_id)
            await repo.add_message(
                session,
                conversation,
                role="assistant",
                content=answer,
                provider=reply.provider,
                model=reply.model,
                wa_message_id=sent_ids[0] if sent_ids else None,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                latency_ms=reply.latency_ms,
            )
            await self.conversations.maybe_summarize(session, conversation)

        logger.info(
            "Answered %s via %s/%s in %sms",
            incoming.from_number,
            reply.provider,
            reply.model,
            reply.latency_ms,
        )

    # ------------------------------------------------------------- internal
    async def _reply(self, incoming: IncomingMessage, text: str) -> list[str]:
        """Send text back, swallowing send errors so the worker keeps going."""
        try:
            return await self.whatsapp.send_text(incoming.from_number, text)
        except WhatsAppError as exc:
            logger.error("Failed to send reply to %s: %s", incoming.from_number, exc)
            return []


__all__ = ["MessageHandler", "PreparedInput"]
