"""Conversation memory: context window plus automatic summarization."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db import repo
from app.db.models import Conversation, Message
from app.llm.base import ChatTurn, ProviderError
from app.llm.registry import ProviderRegistry

logger = logging.getLogger(__name__)

# Portuguese on purpose: the summary is injected into a Portuguese prompt.
SUMMARY_INSTRUCTION = (
    "Resuma a conversa abaixo em no máximo 150 palavras, em português. "
    "Preserve fatos, nomes, números, decisões e pendências; descarte cortesias. "
    "Escreva em tópicos curtos, sem introdução."
)


class ConversationService:
    """Builds the prompt for a thread and keeps its history from growing forever."""

    def __init__(self, registry: ProviderRegistry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings

    # ---------------------------------------------------------- base prompt
    def system_prompt(self, conversation: Conversation) -> str:
        """Persona (or the global default) + who is talking + rolling summary."""
        parts = [conversation.persona or self.settings.system_prompt]

        if conversation.display_name:
            parts.append(f"O nome de quem está falando com você é {conversation.display_name}.")
        if conversation.summary:
            parts.append(
                "Resumo do que já foi conversado antes (use como memória, "
                f"não repita sem necessidade):\n{conversation.summary}"
            )
        return "\n\n".join(parts)

    # -------------------------------------------------------------- history
    async def history_turns(
        self, session: AsyncSession, conversation: Conversation
    ) -> list[ChatTurn]:
        """Recent turns as ChatTurn, skipping whatever the summary already covers.

        Attachments are not replayed: only the text of past turns goes back in,
        which keeps token cost flat over a long thread.
        """
        messages: list[Message] = await repo.recent_messages(
            session,
            conversation.id,
            limit=self.settings.history_max_messages,
            after_id=conversation.summarized_until_id or 0,
        )
        return [
            ChatTurn(role="assistant" if m.role == "assistant" else "user", text=m.content)
            for m in messages
            if m.content
        ]

    # -------------------------------------------------------------- summary
    async def maybe_summarize(self, session: AsyncSession, conversation: Conversation) -> bool:
        """Compress the old part of the history once it gets long.

        Keeps the last HISTORY_MAX_MESSAGES turns verbatim and folds everything
        older into `conversation.summary`. Returns True when it summarized.
        """
        threshold = self.settings.summarize_after_messages
        if threshold <= 0:  # feature disabled
            return False

        pending = await repo.count_messages(
            session, conversation.id, after_id=conversation.summarized_until_id or 0
        )
        if pending < threshold:
            return False

        keep = self.settings.history_max_messages
        tail = await repo.recent_messages(session, conversation.id, limit=keep)
        if not tail:
            return False
        cutoff_id = tail[0].id - 1  # everything before the kept tail

        older = await repo.messages_between(
            session, conversation.id, conversation.summarized_until_id or 0, cutoff_id
        )
        if not older:
            return False

        transcript = "\n".join(
            f"{'Usuário' if m.role == 'user' else 'Assistente'}: {m.content}"
            for m in older
            if m.content
        )
        prompt = SUMMARY_INSTRUCTION
        if conversation.summary:
            # Merge with the previous summary so nothing is lost across rounds.
            prompt += (
                "\n\nJá existe este resumo anterior — integre os dois em um só:\n"
                f"{conversation.summary}"
            )

        try:
            reply = await self.registry.chat(
                [ChatTurn(role="user", text=f"{prompt}\n\n---\n{transcript}")],
                system="Você resume conversas de forma factual e enxuta.",
                provider=conversation.provider,
                max_tokens=400,
                temperature=0.2,
            )
        except ProviderError as exc:
            # Not fatal: try again after the next message.
            logger.warning("Could not summarize conversation %s: %s", conversation.wa_id, exc)
            return False

        conversation.summary = reply.text
        conversation.summarized_until_id = cutoff_id
        logger.info(
            "Conversation %s summarized (%d messages compressed)", conversation.wa_id, len(older)
        )
        return True
