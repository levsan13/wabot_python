"""Slash commands typed inside WhatsApp (/modelo, /reset, ...).

Command names and replies are Portuguese because users type and read them;
comments and identifiers are English.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db import repo
from app.db.models import Conversation
from app.llm.registry import ProviderRegistry

# Suggestions shown by /modelos — switching accepts any id the provider knows.
KNOWN_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
    "anthropic": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"],
    "gemini": ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-pro"],
}

HELP_TEXT = """*Comandos disponíveis*

/ajuda — mostra esta lista
/status — provedor, modelo e tamanho do histórico
/provedor _nome_ — troca entre openai, anthropic e gemini
/modelo _id_ — fixa um modelo (ex.: /modelo claude-sonnet-5)
/modelos — sugestões de modelos por provedor
/persona _texto_ — define como devo me comportar
/persona limpar — volta ao comportamento padrão
/reset — apaga o histórico desta conversa

Também entendo *áudio* (transcrevo), *imagens* e *PDF* — é só mandar."""


@dataclass(slots=True)
class CommandOutcome:
    """Whether the text was a command, and what to answer."""

    handled: bool
    reply: str | None = None


class CommandService:
    """Resolves commands without ever calling an LLM."""

    def __init__(self, registry: ProviderRegistry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings

    @staticmethod
    def is_command(text: str) -> bool:
        return text.strip().startswith("/")

    async def execute(
        self, session: AsyncSession, conversation: Conversation, text: str
    ) -> CommandOutcome:
        """Dispatch "/name argument" to the matching handler."""
        raw = text.strip()
        head, _, argument = raw.partition(" ")
        command = head.lower().lstrip("/")
        argument = argument.strip()

        # Portuguese names with English aliases, so both work.
        handler = {
            "ajuda": self._help,
            "help": self._help,
            "start": self._help,
            "comandos": self._help,
            "status": self._status,
            "info": self._status,
            "provedor": self._provider,
            "provider": self._provider,
            "modelo": self._model,
            "model": self._model,
            "modelos": self._models,
            "models": self._models,
            "persona": self._persona,
            "reset": self._reset,
            "limpar": self._reset,
        }.get(command)

        if handler is None:
            return CommandOutcome(
                handled=True,
                reply=f"Não conheço o comando */{command}*. Digite /ajuda para ver a lista.",
            )
        return await handler(session, conversation, argument)

    # ------------------------------------------------------------- handlers
    async def _help(self, *_args) -> CommandOutcome:
        return CommandOutcome(handled=True, reply=HELP_TEXT)

    async def _status(
        self, session: AsyncSession, conversation: Conversation, _argument: str
    ) -> CommandOutcome:
        """Show the effective provider/model and how big the context is."""
        provider = self.registry.resolve(conversation.provider)
        model = conversation.model or self.settings.model_for(provider)
        pending = await repo.count_messages(
            session, conversation.id, after_id=conversation.summarized_until_id or 0
        )
        lines = [
            "*Status*",
            f"Provedor: *{provider}*" + ("" if conversation.provider else " (padrão)"),
            f"Modelo: *{model}*" + ("" if conversation.model else " (padrão)"),
            f"Provedores configurados: {', '.join(self.registry.available()) or 'nenhum'}",
            f"Mensagens no contexto: {pending}",
            f"Resumo de memória: {'sim' if conversation.summary else 'não'}",
            f"Persona personalizada: {'sim' if conversation.persona else 'não'}",
        ]
        return CommandOutcome(handled=True, reply="\n".join(lines))

    async def _provider(
        self, _session: AsyncSession, conversation: Conversation, argument: str
    ) -> CommandOutcome:
        """Switch provider for this thread only."""
        available = self.registry.available()
        if not argument:
            current = conversation.provider or f"{self.settings.default_provider} (padrão)"
            return CommandOutcome(
                handled=True,
                reply=(
                    f"Provedor atual: *{current}*\n"
                    f"Disponíveis: {', '.join(available) or 'nenhum configurado'}\n"
                    "Use: /provedor openai"
                ),
            )

        choice = argument.lower().strip()
        if choice in ("padrao", "padrão", "default", "auto"):
            conversation.provider = None
            conversation.model = None
            return CommandOutcome(
                handled=True, reply="Voltei para o provedor e o modelo padrão. ✅"
            )
        if choice not in available:
            return CommandOutcome(
                handled=True,
                reply=(
                    f"Não tenho *{choice}* configurado. "
                    f"Disponíveis: {', '.join(available) or 'nenhum'}."
                ),
            )

        conversation.provider = choice
        # A model id from the previous provider is meaningless for the new one.
        conversation.model = None
        return CommandOutcome(
            handled=True,
            reply=f"Pronto, agora respondo pelo *{choice}* "
            f"(modelo {self.settings.model_for(choice)}).",
        )

    async def _model(
        self, _session: AsyncSession, conversation: Conversation, argument: str
    ) -> CommandOutcome:
        """Pin a model id. Not validated here — the provider is the judge."""
        provider = self.registry.resolve(conversation.provider)
        if not argument:
            current = conversation.model or f"{self.settings.model_for(provider)} (padrão)"
            return CommandOutcome(
                handled=True,
                reply=f"Modelo atual: *{current}*\nUse: /modelo {KNOWN_MODELS[provider][0]}",
            )
        if argument.lower() in ("padrao", "padrão", "default", "auto"):
            conversation.model = None
            return CommandOutcome(handled=True, reply="Voltei para o modelo padrão. ✅")

        conversation.model = argument
        return CommandOutcome(
            handled=True,
            reply=f"Modelo definido como *{argument}* (provedor {provider}). "
            "Se o nome estiver errado, o pedido vai falhar — use /modelos para ver sugestões.",
        )

    async def _models(self, *_args) -> CommandOutcome:
        blocks = ["*Modelos sugeridos*"]
        for provider, models in KNOWN_MODELS.items():
            badge = "" if self.registry.is_available(provider) else " _(sem API key)_"
            blocks.append(f"\n*{provider}*{badge}")
            blocks.extend(f"• {m}" for m in models)
        blocks.append("\nQualquer id aceito pela API do provedor funciona.")
        return CommandOutcome(handled=True, reply="\n".join(blocks))

    async def _persona(
        self, _session: AsyncSession, conversation: Conversation, argument: str
    ) -> CommandOutcome:
        """Replace the system prompt for this thread."""
        if not argument:
            current = conversation.persona or "(padrão)"
            return CommandOutcome(
                handled=True,
                reply=f"*Persona atual:*\n{current}\n\nUse: /persona seja formal e cite fontes",
            )
        if argument.lower() in ("limpar", "reset", "padrao", "padrão"):
            conversation.persona = None
            return CommandOutcome(handled=True, reply="Persona removida — voltei ao padrão. ✅")

        conversation.persona = argument
        return CommandOutcome(handled=True, reply="Persona atualizada. ✅")

    async def _reset(
        self, session: AsyncSession, conversation: Conversation, _argument: str
    ) -> CommandOutcome:
        """Wipe the history and the summary for this thread."""
        removed = await repo.reset_conversation(session, conversation)
        return CommandOutcome(
            handled=True,
            reply=f"Histórico apagado ({removed} mensagens). Começamos do zero. 🧹",
        )
