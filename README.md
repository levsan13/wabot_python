# wabot_python

A Python (FastAPI) API that connects the **WhatsApp Cloud API** to three LLM providers —
**OpenAI**, **Anthropic** and **Google Gemini** — with runtime provider switching, automatic
fallback, conversation memory, voice-note transcription and image/PDF understanding.

```
WhatsApp  ──►  POST /webhook  ──►  queue  ──►  handler  ──►  OpenAI | Anthropic | Gemini
   ▲              (200 right away)              │
   └──────────────  send_text  ◄────────────────┘
```

> The bot itself speaks Portuguese (system prompt, `/ajuda`, `/provedor`, user-facing replies).
> Code, comments and this documentation are in English. Changing the bot's language means
> editing `DEFAULT_SYSTEM_PROMPT` in `app/config.py` and the strings in `app/services/`.

---

## Contents

- [What it does](#what-it-does)
- [Layout](#layout)
- [Install](#install)
- [Setting up WhatsApp on Meta](#setting-up-whatsapp-on-meta)
- [Running](#running)
- [In-chat commands](#in-chat-commands)
- [HTTP endpoints](#http-endpoints)
- [How it works inside](#how-it-works-inside)
- [Supported models](#supported-models)
- [Tests](#tests)
- [Before going to production](#before-going-to-production)

---

## What it does

| Feature | Detail |
|---|---|
| **3 providers** | OpenAI, Anthropic and Gemini behind one interface (`app/llm/base.py`) |
| **Fallback** | If the primary provider fails (quota, timeout, 5xx), the next one in line answers |
| **Memory** | Per-contact history in SQLite; whatever falls out of the window becomes a summary |
| **Audio** | Voice notes are downloaded and transcribed before reaching the model |
| **Vision** | Images and PDFs are passed natively to the multimodal model |
| **Commands** | `/provedor`, `/modelo`, `/persona`, `/reset`… straight from WhatsApp |
| **Security** | `X-Hub-Signature-256` validated, number allow-list, API key on the `/api` routes |
| **Robustness** | Webhook deduplication, background queue, splitting of messages over 4096 chars |

## Layout

```
wabot_python/
├── app/
│   ├── main.py               # FastAPI + lifespan (database, providers, queue)
│   ├── config.py             # every env var, with defaults
│   ├── api/
│   │   ├── webhook.py        # GET (verification) and POST (events) from Meta
│   │   ├── admin.py          # /api/chat, /api/messages, /api/conversations
│   │   ├── health.py         # /health diagnostics
│   │   └── deps.py           # object graph + API key guard
│   ├── llm/
│   │   ├── base.py           # ChatTurn, Attachment, LLMReply, LLMProvider
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── gemini_provider.py
│   │   └── registry.py       # provider choice, fallback chain, transcription
│   ├── whatsapp/
│   │   ├── client.py         # sending, "typing…", media download
│   │   ├── schemas.py        # Meta payload -> IncomingMessage
│   │   └── security.py       # webhook HMAC
│   ├── services/
│   │   ├── handler.py        # orchestrates the answer
│   │   ├── dispatcher.py     # async queue with a per-number lock
│   │   ├── conversation.py   # context window + summarization
│   │   ├── media.py          # audio/image/PDF -> prompt
│   │   └── commands.py       # in-chat commands
│   └── db/                   # async SQLAlchemy (models, repo, engine)
├── tests/                    # 43 tests, no live API calls
├── run.py                    # python run.py --reload
└── .env.example
```

## Install

```bash
# 1) virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

# 2) dependencies
pip install -r requirements.txt

# 3) configuration
copy .env.example .env           # Windows  (cp on Linux/macOS)
```

Requires **Python 3.10+**.

## Setting up WhatsApp on Meta

1. **Create the app** at [developers.facebook.com](https://developers.facebook.com/apps) →
   *Create app* → type **Business** → add the **WhatsApp** product.
2. Under **WhatsApp → API Setup** you get a **test number** and a temporary token (24 h).
   Note the **Phone number ID** → `WA_PHONE_NUMBER_ID`.
3. Add your own phone under *To* (recipient numbers) so you can test.
4. **Permanent token**: *Business settings → System users* → create an admin system user,
   grant it access to the app and generate a token with the
   `whatsapp_business_messaging` and `whatsapp_business_management` permissions →
   `WA_ACCESS_TOKEN`.
5. **App secret**: *App settings → Basic → App secret* → `WA_APP_SECRET`.
6. **Expose your local server** (Meta requires public HTTPS):

   ```bash
   ngrok http 8000
   # or: cloudflared tunnel --url http://localhost:8000
   ```

7. **Webhook**: *WhatsApp → Configuration → Webhook → Edit*
   - Callback URL: `https://YOUR-TUNNEL/webhook`
   - Verify token: the same value as `WA_VERIFY_TOKEN`
   - Click **Verify and save** (the app must be running)
   - Under **Manage**, subscribe to the **`messages`** field ✅

> Meta only delivers webhooks over HTTPS with a valid certificate — `ngrok` or `cloudflared`
> handle that in development.

Fill in at least one LLM key in `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or
`GEMINI_API_KEY`). `/health` reports which ones were picked up.

## Running

```bash
python run.py --reload
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Check the diagnostics:

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "providers": {"default": "openai", "available": ["openai", "gemini"]},
  "transcription": {"provider": "openai", "ready": true},
  "queue_pending": 0
}
```

Send "oi" to the test number on WhatsApp and watch the log.

### Docker

```bash
docker build -t wabot .
docker run --env-file .env -p 8000:8000 -v "%cd%/data:/app/data" wabot
```

## In-chat commands

| Command | What it does |
|---|---|
| `/ajuda` | Lists the commands |
| `/status` | Provider, model, context size, whether a summary exists |
| `/provedor gemini` | Switches provider for this thread (`/provedor padrão` restores the default) |
| `/modelo claude-opus-5` | Pins a specific model |
| `/modelos` | Model suggestions per provider |
| `/persona responda como um professor` | Replaces the system prompt for this thread |
| `/persona limpar` | Restores the default prompt |
| `/reset` | Wipes the conversation history |

Commands cost no tokens: they are resolved before any LLM call.

## HTTP endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/webhook` | Meta's verification handshake |
| `POST` | `/webhook` | Receives events (validates signature, queues, answers 200) |
| `GET` | `/health` | Diagnostics |
| `POST` | `/api/chat` | Talk to the LLM without WhatsApp |
| `POST` | `/api/messages` | Send a message from the bot's number |
| `GET` | `/api/conversations` | List conversations |
| `GET` | `/api/conversations/{wa_id}/messages` | History of one conversation |
| `DELETE` | `/api/conversations/{wa_id}/history` | Clear the history |

Interactive docs at `http://localhost:8000/docs`.
The `/api/*` routes require an `X-API-Key` header when `ADMIN_API_KEY` is set.

```bash
# test provider routing without WhatsApp
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" -H "X-API-Key: your-key" \
  -d '{"provider":"anthropic","messages":[{"role":"user","content":"explain entropy in 2 sentences"}]}'
```

## How it works inside

1. **`POST /webhook`** validates the HMAC over the raw body, turns Meta's JSON into
   `IncomingMessage` objects and **answers 200 immediately** — Meta redelivers an event when
   the endpoint is slow, which is what makes most bots answer twice.
2. The **dispatcher** processes messages in the background with N workers and one
   `asyncio.Lock` per number, so a single person's messages stay in order.
3. The **handler** records the `wamid` in `processed_events` (deduplication), marks the
   message as read and turns on the typing indicator.
4. Commands answer straight away. Otherwise the **media service** resolves the attachment:
   audio → transcription, image/PDF → `Attachment`, txt/json → inlined in the prompt.
5. The **conversation service** assembles the prompt: persona + summary + last N turns.
6. The **registry** calls the thread's provider; on `ProviderError` it walks down
   `PROVIDER_FALLBACK_ORDER`.
7. The answer is split into ~3,800-character chunks and sent; everything is persisted.
8. Past `SUMMARIZE_AFTER_MESSAGES`, the old part of the history is compressed into a short
   summary — the context stops growing without the conversation losing the thread.

## Supported models

Any id the provider's API accepts works. Defaults from `.env.example`:

| Provider | Default | Others | Audio | Image | PDF |
|---|---|---|---|---|---|
| OpenAI | `gpt-5.6-terra` | `gpt-6-astra`, `gpt-5.6-sol`, `gpt-5.6-luna` | ✅ (`gpt-transcribe`) | ✅ | ✅ |
| Anthropic | `claude-sonnet-5` | `claude-opus-5`, `claude-haiku-4-5-20251001` | ❌ (another provider transcribes) | ✅ | ✅ |
| Gemini | `gemini-3.5-flash` | `gemini-3.8-flash`, `gemini-2.5-pro` | ✅ (native) | ✅ | ✅ |

`OPENAI_BASE_URL` points the OpenAI adapter at any compatible endpoint — LM Studio, vLLM,
Ollama — if you would rather run a local model instead.

## Tests

```bash
pytest -q          # 43 tests, none of them hit an external API
ruff check .
```

They cover: parsing of Meta's real payloads (text, image, audio, document, button, status,
unknown fields), HMAC validation, long-message splitting, deduplication, history, commands,
transcription, provider failure, the allow-list and the HTTP routes.

## Before going to production

- **`WA_APP_SECRET` is mandatory.** Without it, anyone who finds your URL can make the bot
  answer — and spend your tokens.
- **24-hour window**: you may only send free-form text within 24 h of the user's last
  message. Outside it, only approved *templates* work — `/api/messages` will fail.
- **Use a permanent token** (system user); the dashboard token expires in 24 h.
- **`WA_ALLOWED_NUMBERS`** restricts who the bot answers while you are testing.
- SQLite is fine for a personal or team bot. For higher volume, point `DATABASE_URL` at
  Postgres (`postgresql+asyncpg://…`) — the code is already async and storage-agnostic.
- Run it behind a TLS-terminating proxy and keep `ADMIN_API_KEY` set.
