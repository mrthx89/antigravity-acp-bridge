<div align="center">

# Antigravity ACP Bridge

**OpenAI-compatible bridge over Google Antigravity ACP (`agy acp --stdio`)**

*Trademark: YHDev Bekerja dengan Hati @ 2026*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)](#requirements)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Features](#features) · [Quick Start](#quick-start) · [Models](#available-models) · [Configuration](#configuration) · [Docs](#how-it-works)

</div>

---

## What is this?

A lightweight, zero-dependency **OpenAI-compatible bridge** that connects your favorite tools (Hermes, Aider, Continue.dev, Cursor, OpenAI SDK) to Google Antigravity models using the official **`agy` CLI (`agy acp --stdio`)** by default.

Every request runs through Google's official binary with its own genuine client harness fingerprint, completely avoiding the dangerous OAuth token scraping / HTTP replay that gets accounts banned on third-party reverse proxies.

```
Your Client (OpenAI format)  ──HTTP──>  Antigravity ACP Bridge (:8877)  ──JSON-RPC/stdio──>  agy acp  ──>  Google AI Pro
```

### Why use this?

- **Real Google Harness Fingerprint** — drives `agy acp --stdio` directly, zero risk of ToS ban from foreign HTTP client spoofing
- **Multimodal / Vision Support** — pass images (`image_url` data URI or HTTP link) directly from OpenAI clients
- **Banner Protection** — intercepts Google ToS & `VALIDATION_REQUIRED` banners at the stream head so ban text never leaks as an answer
- **Cross-platform** — tested on **Windows** and **Linux** (includes systemd user unit)
- **External-access ready** — built-in Bearer / `x-api-key` authentication, CORS headers, and `--max-concurrency` worker limit
- **Zero dependencies** — pure Python 3.10+ stdlib, no `pip install` needed
- **Streaming** — SSE streaming support (`"stream": true`) for real-time token generation
- **Trademark** — *YHDev Bekerja dengan Hati @ 2026*

---

## Features

- **OpenAI-compatible** `POST /v1/chat/completions` (streaming + non-streaming, function calling, vision) and `GET /v1/models`
- **Persistent ACP worker pool (`agy-cli`, default)** — keeps `agy acp --stdio` warm with `--disable-telemetry`, `--include-default-customizations=false`, and `--cli_log_file` suppressed
- **Google Ban & Verification Guard** — buffers initial stream tokens to intercept `Terms of Service` or `Verification Required` banners and extracts Google verification links
- **Concurrency & auth protection** — `--max-concurrency` guard + mandatory `--api-key` when binding non-localhost hosts
- **Legacy `oauth` mode (`--backend oauth`)** — optional direct Cloud Code Assist API + native Gemini passthrough (`POST /v1beta/models/{model}:generateContent`)
- **Zero pip dependencies** — Python 3.10+ standard library only
- **systemd service** included — run as a background service on Linux (`antigravity-acp-bridge.service`)

---

| Friendly Name                      | Backend Model                      | Thinking   | Type       |
|--------------------------------------|-------------------------------------|-------------|-------------|
| `gemini-4-hydrogen-high` | `gemini-4-hydrogen-high` | High      | Gemini     |
| `gemini-4-hydrogen-xhigh` | `gemini-4-hydrogen-xhigh` | High      | Gemini     |
| `gemini-4-hydrogen-medium` | `gemini-4-hydrogen-medium` | Medium    | Gemini     |
| `gemini-4-hydrogen-low` | `gemini-4-hydrogen-low` | Low       | Gemini     |
| `gemini-4-argon-high` | `gemini-4-argon-high` | High      | Gemini     |
| `gemini-4-argon-medium` | `gemini-4-argon-medium` | Medium    | Gemini     |
| `gemini-4-argon-low` | `gemini-4-argon-low` | Low       | Gemini     |
| `gemini-pro-orchestrator-g1-pro` | `gemini-pro-orchestrator-g1-pro` | High      | Gemini     |
| `gemini-3.8-flash-high` | `gemini-3.8-flash-high` | High      | Gemini     |
| `gemini-3.8-flash-medium` | `gemini-3.8-flash-medium` | Medium    | Gemini     |
| `gemini-3.8-flash-low` | `gemini-3.8-flash-low` | Low       | Gemini     |
| `gpt-oss-120b-medium` | `gpt-oss-120b-medium` | Medium    | GPT-OSS    |
| `claude-sonnet-4-6` | `claude-sonnet-4-6` | Enabled   | Claude     |
| `claude-sonnet-5-5-low` | `claude-sonnet-5-5-low` | Low       | Claude     |
| `claude-sonnet-5-5-medium` | `claude-sonnet-5-5-medium` | Medium    | Claude     |
| `claude-sonnet-5-5-high` | `claude-sonnet-5-5-high` | High      | Claude     |
| `claude-opus-4-6-thinking` | `claude-opus-4-6-thinking` | Enabled   | Claude     |
| `claude-opus-5-5-low` | `claude-opus-5-5-low` | Low       | Claude     |
| `claude-opus-5-5-medium` | `claude-opus-5-5-medium` | Medium    | Claude     |
| `claude-opus-5-5-high` | `claude-opus-5-5-high` | High      | Claude     |

> **Note:** Always use the **friendly name** in your client — the bridge translates it to the backend model id automatically. Whether a model actually **runs** depends on your Antigravity package/access tier; any may return `404` until your account provisions it.

> **Recommended:** `gemini-3.8-flash-high` — Gemini 3.8 Flash with high reasoning (matches Antigravity CLI 1.5.10).

> **Aliases:** `gemini-3.8-flash`, `claude-sonnet-4.6`, `claude-opus-4.6`, and `gpt-oss-120b` are also accepted as short forms of their full model ids.
---

## Quick Start

### Prerequisites

1. **Python 3.10+** — check with `python3 --version`
2. **Antigravity CLI** — install and authenticate (one-time):

```bash
# Install agy (macOS/Linux)
curl -fsSL https://storage.googleapis.com/antigravity-releases/install.sh | bash

# Authenticate (opens browser, sign in with Google)
agy
```

> Windows: download from [antigravity.google](https://antigravity.google/)

### Install & Run

```bash
git clone https://github.com/mrthx89/antigravity-acp-bridge.git
cd antigravity-acp-bridge
```

**Recommended (local & conservative concurrency for personal Google AI Pro usage):**

```bash
# Windows (PowerShell) or Linux/macOS (use python3 on Linux/macOS)
python antigravity_acp_bridge.py --host 127.0.0.1 --port 8877 --max-concurrency 1 --api-key "YOUR_LONG_SECRET"
```

*(Note: `python antigravity_acp_bridge.py ...` remains supported as a backward-compatible alias).*

**Expose on LAN (other devices on your network) or behind a reverse proxy/tunnel:**

```bash
python antigravity_acp_bridge.py --host 0.0.0.0 --port 8877 --max-concurrency 1 --api-key "YOUR_LONG_SECRET"
```

> **Public internet access:** Keep `--host 127.0.0.1` with `--api-key` and expose it through an HTTPS tunnel such as `cloudflared tunnel --url http://127.0.0.1:8877` or Tailscale so your API key is encrypted with TLS.

**Verify it is running:**

```bash
# Health check (public)
curl http://127.0.0.1:8877/health
# → {"status": "ok", "service": "antigravity-acp-bridge", "trademark": "YHDev Bekerja dengan Hati @ 2026"}

# List models (authenticated when --api-key is set)
curl http://127.0.0.1:8877/v1/models -H "Authorization: Bearer YOUR_LONG_SECRET"
```

### Connect Your Tool

Any OpenAI-compatible client works. Set base URL to `http://127.0.0.1:8877/v1` (or your LAN/tunnel URL) and set `api_key` to the value passed to `--api-key` (or any string if running localhost-only without `--api-key`).

<details>
<summary><b>OpenAI Python SDK</b></summary>

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8877/v1",
    api_key="YOUR_LONG_SECRET",
)

response = client.chat.completions.create(
    model="gemini-3.8-flash-high",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```
</details>

<details>
<summary><b>curl</b></summary>

```bash
curl http://127.0.0.1:8877/v1/chat/completions \
  -H "Authorization: Bearer YOUR_LONG_SECRET" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.8-flash-high",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
</details>

<details>
<summary><b>Aider</b></summary>

```bash
export OPENAI_API_BASE=http://127.0.0.1:8877/v1
export OPENAI_API_KEY=antigravity
aider --model gemini-3.8-flash-high
```
</details>

<details>
<summary><b>Continue.dev (VS Code)</b></summary>

In `~/.continue/config.json`:
```json
{
  "models": [{
    "title": "Antigravity",
    "provider": "openai",
    "model": "gemini-3.8-flash-high",
    "apiBase": "http://127.0.0.1:8877/v1",
    "apiKey": "antigravity"
  }]
}
```
</details>

<details>
<summary><b>Hermes Agent</b></summary>

```bash
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8877/v1
hermes config set model.api_key antigravity
hermes config set model.default gemini-3.8-flash-high
```
</details>

<details>
<summary><b>Claude Code</b></summary>

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8877/v1
claude  # uses claude-sonnet-4.6 through the bridge
```
</details>

<details>
<summary><b>LangChain</b></summary>

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8877/v1",
    api_key="antigravity",
    model="gemini-3.8-flash-high",
)
```
</details>

---

## AI-Assisted Setup 🤖

**Using an AI assistant (Claude, GPT, Gemini, etc.) to set this up?**

This repo includes [`AGENTS.md`](AGENTS.md) — a complete instruction file for AI agents. Just tell your AI assistant:

> "Help me install and configure the antigravity-acp-bridge from https://github.com/mrthx89/antigravity-acp-bridge"

The AI will read `AGENTS.md` and guide you through every step: prerequisites check, installation, tool configuration, and troubleshooting.

---

## Configuration

### CLI Arguments

```
python3 antigravity_acp_bridge.py [--port PORT] [--host HOST] [--backend {agy-cli,oauth}] [--api-key KEY] [--cors-origin ORIGINS]

  --port PORT                  Port to listen on (default: 8877)
  --host HOST                  Host to bind to (default: 127.0.0.1)
  --backend {agy-cli,oauth}    Execution backend: agy-cli (default, persistent official `agy acp` workers) or oauth (legacy direct API)
  --agy-bin PATH               Path/command for `agy` when using --backend agy-cli (default: agy)
  --agy-cwd DIR                Working directory for agy-cli calls (default: isolated temp dir)
  --max-concurrency N          Max concurrent agy-cli calls (default: 2); 0 or negative = unlimited
  --worker-idle-timeout SECS   Close an idle agy worker after SECS (default: 300; 0 = never reap)
  --api-key KEY                Require Bearer / x-api-key auth on API routes
  --allow-unauthenticated      Allow non-localhost binding without --api-key (unsafe)
  --cors-origin ORIGINS        CORS Allow-Origin header or comma-separated list (default: *)
  --log-level LEVEL            DEBUG, INFO, WARNING, ERROR (default: INFO)
```

> **Module entry point:** You can also run `python3 -m antigravity_acp_bridge` or install with `pip install -e .` and run `antigravity-acp-bridge`.

### Run as a systemd Service (Linux)

```bash
# Clone to permanent location
git clone https://github.com/mrthx89/antigravity-acp-bridge.git ~/antigravity-acp-bridge

# Install service
mkdir -p ~/.config/systemd/user
cp ~/antigravity-acp-bridge/antigravity-acp-bridge.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable antigravity-acp-bridge
systemctl --user start antigravity-acp-bridge

# Check status
systemctl --user status antigravity-acp-bridge
```

### Environment Variables

| Variable                     | Default  | Description                                  |
|------------------------------|----------|----------------------------------------------|
| `ANTIGRAVITY_BRIDGE_HOST`     | `127.0.0.1` | Default bind host                         |
| `ANTIGRAVITY_BRIDGE_PORT`     | `8877`   | Default bind port                            |
| `ANTIGRAVITY_BACKEND`        | `agy-cli`| Backend mode: `agy-cli` (official) or `oauth`|
| `ANTIGRAVITY_AGY_BIN`        | `agy`    | `agy` executable for `agy-cli` backend       |
| `ANTIGRAVITY_AGY_CWD`        | temp dir | Working directory for `agy-cli` calls        |
| `ANTIGRAVITY_MAX_CONCURRENCY`| `2`      | Max concurrent `agy-cli` workers (`0`=unlim) |
| `ANTIGRAVITY_BRIDGE_API_KEY`  | unset    | API key required via Bearer or `x-api-key`   |
| `ANTIGRAVITY_CORS_ORIGINS`   | `*`      | Allowed CORS origins (`none` to disable)     |
| `ANTIGRAVITY_TOKEN_FILE`     | `~/.gemini/antigravity-cli/antigravity-oauth-token` | Override OAuth token file path |
| `ANTIGRAVITY_CLIENT_ID`      | built-in | Override OAuth client ID                     |
| `ANTIGRAVITY_CLIENT_SECRET`  | built-in | Override OAuth client secret                 |
| `ANTIGRAVITY_PROJECT_ID`     | auto     | Override project ID (auto-discovered)        |

The default client credentials are the same values embedded in every Antigravity CLI binary — you only need these env vars if you're using a custom build.

---

## Two API Endpoints

### 1. OpenAI Chat Completions

```
POST /v1/chat/completions
```

Standard OpenAI format with messages, tools, and streaming. Use this for any OpenAI-compatible client.

### 2. Native Gemini Passthrough

```
POST /v1beta/models/{model}:generateContent
```

Accepts the exact same format as Google's Generative Language API. Use this for:

- **Image inputs** (`inline_data` with base64-encoded images)
- **Structured output** (`responseSchema` for enforced JSON schema)
- **Thinking control** (`thinkingConfig`)

<details>
<summary><b>Structured output example</b></summary>

```bash
curl http://127.0.0.1:8877/v1beta/models/gemini-3.8-flash-high:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "Extract: Ahmed is 30 years old"}]}],
    "generationConfig": {
      "temperature": 0,
      "responseMimeType": "application/json",
      "responseSchema": {
        "type": "OBJECT",
        "properties": {
          "name": {"type": "STRING"},
          "age": {"type": "NUMBER"}
        },
        "required": ["name", "age"]
      }
    }
  }'
```
</details>

<details>
<summary><b>Image input example</b></summary>

```bash
# Base64-encode an image and send it
B64=$(base64 -w0 photo.jpg)

curl http://127.0.0.1:8877/v1beta/models/gemini-3.8-flash-high:generateContent \
  -H "Content-Type: application/json" \
  -d "{
    \"contents\": [{
      \"parts\": [
        {\"text\": \"What is in this image?\"},
        {\"inline_data\": {\"mime_type\": \"image/jpeg\", \"data\": \"$B64\"}}
      ]
    }]
  }"
```
</details>

---

## Management

```bash
# Health check
curl http://127.0.0.1:8877/health

# List models
curl http://127.0.0.1:8877/v1/models

# View logs (systemd)
journalctl --user -u antigravity-acp-bridge -f

# Restart / stop
systemctl --user restart antigravity-acp-bridge
systemctl --user stop antigravity-acp-bridge

# Re-authenticate when token expires
agy && systemctl --user restart antigravity-acp-bridge
```

---

## How It Works

### Default Mode (`--backend agy-cli`)

```
Your Client (OpenAI format)
    │
    ▼
Antigravity ACP Bridge (localhost:8877)
    │  ┌─ Verifies Bearer / x-api-key (when --api-key is set)
    │  ├─ Queues via --max-concurrency to avoid request bursts
    │  ├─ Reuses warm `agy acp --stdio` workers (JSON-RPC)
    │  └─ Maps OpenAI messages ↔ ACP session/prompt & SSE chunks
    │
    ▼
Official `agy` CLI → Gemini / Claude models → OpenAI format → Your client
```

### Legacy Mode (`--backend oauth`)

```
Your Client (OpenAI format)
    │
    ▼
Antigravity ACP Bridge (localhost:8877)
    │  ┌─ Translates OpenAI messages → Gemini contents
    │  ├─ Converts OpenAI tools → Gemini functionDeclarations
    │  ├─ Manages OAuth token (auto-refresh)
    │  ├─ Maps friendly model names → backend model names
    │  └─ Wraps in Antigravity envelope {project, model, request}
    │
    ▼
daily-cloudcode-pa.googleapis.com
    │
    ▼
Gemini / Claude models → Response unwrapped → OpenAI format → Your client
```

<details>
<summary><b>Technical details</b></summary>

1. Reads OAuth token from `~/.gemini/antigravity-cli/antigravity-oauth-token`
2. Refreshes expired access tokens via `oauth2.googleapis.com/token`
3. Discovers Cloud Code Assist project ID via `v1internal:loadCodeAssist`
4. Wraps each request in the Antigravity envelope: `{project, model, request, requestType, userAgent, requestId}`
5. Forwards to `v1internal:generateContent` or `v1internal:streamGenerateContent?alt=sse`
6. Unwraps the `{response: {...}}` envelope
7. Converts Gemini response → OpenAI format (choices, message, usage)
8. For tool calls, encodes Gemini's `functionCall.id` + `thoughtSignature` into OpenAI `tool_call.id` for round-tripping
</details>

---

## Adding or Updating Models

Edit `MODEL_MAP` in `antigravity_acp_bridge/config.py`:

```python
MODEL_MAP = {
    "gemini-3.8-flash-high": ("gemini-3.8-flash-high", "HIGH"),
    "gemini-3.8-flash-medium": ("gemini-3.8-flash-medium", "MEDIUM"),
    "gemini-3.8-flash-low": ("gemini-3.8-flash-low", "LOW"),
    # "friendly-name": ("backend-name", thinking_level),
    # thinking_level: "LOW", "MEDIUM", "HIGH", or None
}
```

Restart the bridge after editing.

---

## Requirements

- Python 3.10+
- Antigravity CLI (`agy`) installed and authenticated
- No pip dependencies

---

## FAQ

<details>
<summary><b>Is this legal to use?</b></summary>

The bridge uses the same OAuth token and API endpoints that the Antigravity CLI itself uses. It does not bypass authentication — you need a valid Google account with Antigravity access. Multiple similar projects exist publicly on GitHub. However, Google's terms of service may restrict programmatic API access. Use at your own discretion.
</details>

<details>
<summary><b>Does it support streaming?</b></summary>

Yes. SSE streaming (`"stream": true`) on the OpenAI endpoint, and the native Gemini stream endpoint.
</details>

<details>
<summary><b>Can I send images?</b></summary>

Yes, through the native Gemini passthrough endpoint (`/v1beta/models/{model}:generateContent`). Send `inline_data` with base64-encoded images, exactly like Google's API.
</details>

<details>
<summary><b>Is there a rate limit?</b></summary>

Upstream quota is enforced by Google's Antigravity service. Locally, `--backend agy-cli` limits concurrent workers via `--max-concurrency` (default `2`; use `1` for conservative personal usage).

Passing `0` (or any negative value) means **unlimited**: no cap is applied, so every concurrent request may hold its own `agy` worker. Budget for it — each warm worker is a full `agy acp --stdio` process at roughly **230 MB RSS**, and workers that finish a request stay pooled at the historical peak concurrency until the bridge restarts. On a 4 GB host, unlimited plus a burst of 5+ parallel requests will push you into swap. Set a positive number unless the machine has headroom.

The idle reaper keeps that bounded: a worker parked longer than `--worker-idle-timeout`
(default `300` s) is closed and its memory returned, so the pool shrinks back down after a
burst instead of staying at peak forever. The cost is a cold start on the next request
that needs a fresh worker (spawn + ACP session init; measured on this host, a cold request
took 14.4 s versus 16.5 s warm — end-to-end latency is dominated by upstream model time,
so the cold-start penalty is small and noisy). Set `--worker-idle-timeout 0` to keep
workers warm indefinitely.
</details>

<details>
<summary><b>Does the token expire?</b></summary>

The OAuth access token expires (~1 hour). The bridge auto-refreshes it. The refresh token can expire if unused for extended periods — re-run `agy` if that happens.
</details>

<details>
<summary><b>Can I bind to 0.0.0.0?</b></summary>

Yes. When binding to a non-localhost address such as `0.0.0.0`, the bridge requires `--api-key` (or `ANTIGRAVITY_BRIDGE_API_KEY`) by default so unauthorized callers cannot use your Antigravity quota. Pass `--allow-unauthenticated` only on a trusted private network.
</details>

---

## License

[MIT](LICENSE)
