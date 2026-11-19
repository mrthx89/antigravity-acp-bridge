# AGENTS.md — LLM Agent Instructions (Antigravity ACP Bridge)

> **Trademark: YHDev Bekerja dengan Hati @ 2026**
> **If you are an AI assistant (Claude, GPT, Gemini, etc.) helping a user set up or troubleshoot this bridge, read this file first. It contains everything you need to guide the user through installation, configuration, testing, and troubleshooting.**

---

## What This Project Is

A local, zero-dependency OpenAI-compatible bridge over Google Antigravity. By default, it translates OpenAI-compatible API calls into Google's **Agent Client Protocol (ACP)** via persistent `agy acp --stdio` JSON-RPC workers.

Because it runs Google's official `agy` binary locally:
- It preserves Google's authentic client telemetry and harness fingerprint (no ToS bans from foreign HTTP client spoofing).
- It supports multimodal vision (extracts OpenAI `image_url` data URI and remote HTTP images into native ACP image blocks).
- It features an automated stream-head guard that intercepts Google ToS / `VALIDATION_REQUIRED` banners before they can leak into client responses, extracting actionable Google verification links.

The bridge listens on `http://127.0.0.1:8877` by default and exposes:
- `POST /v1/chat/completions` — OpenAI chat format (streaming, non-streaming, vision, tool calls)
- `GET /v1/models` — OpenAI model listing
- `GET /health` — Health check endpoint (`{"status": "ok", "service": "antigravity-acp-bridge", "trademark": "YHDev Bekerja dengan Hati @ 2026"}`)
- `POST /v1beta/models/{model}:generateContent` — Native Gemini passthrough (when running `--backend oauth`)

---

## How to Guide a User Through Setup

### Step 1 — Check Prerequisites

Ask the user or verify via terminal:
1. **Python 3.10+**:
   ```bash
   python3 --version   # Linux/macOS
   python --version    # Windows
   ```
2. **Antigravity CLI (`agy`) installed and in PATH**:
   ```bash
   agy --version
   ```
   If not installed:
   - **macOS/Linux**:
     ```bash
     curl -fsSL https://storage.googleapis.com/antigravity-releases/install.sh | bash
     ```
   - **Windows**: Download from [antigravity.google](https://antigravity.google/).
3. **Authenticated with Google Account**:
   Run `agy` once in terminal and complete Google OAuth in the browser.
   Token location:
   - Linux/macOS: `~/.gemini/antigravity-cli/antigravity-oauth-token`
   - Windows: `%USERPROFILE%\.gemini\antigravity-cli\antigravity-oauth-token`

### Step 2 — Start the Bridge

Navigate to the project directory and start:

```bash
# Windows (PowerShell / Command Prompt)
python antigravity_acp_bridge.py

# Linux / macOS
python3 antigravity_acp_bridge.py
```

*Recommended production/LAN setting with concurrency guard and auth:*
```bash
python antigravity_acp_bridge.py --host 127.0.0.1 --port 8877 --max-concurrency 2 --api-key "YOUR_SECRET_KEY"
```

Verify it is running:
```bash
curl http://127.0.0.1:8877/health
# Expected output:
# {"status": "ok", "service": "antigravity-acp-bridge", "trademark": "YHDev Bekerja dengan Hati @ 2026"}
```

### Step 3 — Configure Their Client Tool

Guide the user to configure their preferred tool:

| Tool / Environment | Base URL | API Key | Recommended Model |
|--------------------|----------|---------|-------------------|
| **OpenAI Python / Node SDK** | `http://127.0.0.1:8877/v1` | Any string (or `--api-key`) | `gemini-3.8-flash-high` |
| **Continue.dev** | `http://127.0.0.1:8877/v1` | Any string (or `--api-key`) | `gemini-3.8-flash-high` |
| **Aider** | `--openai-api-base http://127.0.0.1:8877/v1` | Any string (or `--api-key`) | `gemini-3.8-flash-high` |
| **Hermes Agent** | Configured via `hermes config` | Any string (or `--api-key`) | `gemini-3.8-flash-high` |
| **Cursor / LibreChat** | `http://127.0.0.1:8877/v1` | Any string (or `--api-key`) | `gemini-4-hydrogen-high` |
| **Claude-compatible Tools** | `http://127.0.0.1:8877/v1` | Any string (or `--api-key`) | `claude-sonnet-4-6` / `claude-opus-4.6` |

> *Note on API Key:* When running bound to `127.0.0.1` without `--api-key`, clients can provide any arbitrary string (e.g. `dummy`). If `--api-key` is configured or `--host 0.0.0.0` is used, the client must pass the matching key in `Authorization: Bearer <key>` or `x-api-key: <key>`.

### Step 4 — Verify It Works

Run a quick test request:

```bash
# Non-streaming test
curl http://127.0.0.1:8877/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.8-flash-high",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

```bash
# Streaming test
curl http://127.0.0.1:8877/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.8-flash-high",
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "stream": true
  }'
```

---

## Available Models

The bridge translates client model names to Antigravity backend model IDs and reasoning profiles:

| Model Identifier (Client) | Backend Target | Thinking Level | Family |
|---|---|---|---|
| `gemini-3.8-flash-high` *(Recommended)* | `gemini-3.8-flash-high` | High | Gemini 3.8 |
| `gemini-3.8-flash-medium` | `gemini-3.8-flash-medium` | Medium | Gemini 3.8 |
| `gemini-3.8-flash-low` | `gemini-3.8-flash-low` | Low | Gemini 3.8 |
| `gemini-4-hydrogen-high` | `gemini-4-hydrogen-high` | High | Gemini 4 Hydrogen |
| `gemini-4-hydrogen-xhigh` | `gemini-4-hydrogen-xhigh` | High | Gemini 4 Hydrogen |
| `gemini-4-hydrogen-medium` | `gemini-4-hydrogen-medium` | Medium | Gemini 4 Hydrogen |
| `gemini-4-hydrogen-low` | `gemini-4-hydrogen-low` | Low | Gemini 4 Hydrogen |
| `gemini-4-argon-high` | `gemini-4-argon-high` | High | Gemini 4 Argon |
| `gemini-4-argon-medium` | `gemini-4-argon-medium` | Medium | Gemini 4 Argon |
| `gemini-4-argon-low` | `gemini-4-argon-low` | Low | Gemini 4 Argon |
| `gemini-pro-orchestrator-g1-pro` | `gemini-pro-orchestrator-g1-pro` | High | Gemini Pro Orchestrator |
| `gpt-oss-120b-medium` (or `gpt-oss-120b`) | `gpt-oss-120b-medium` | None | Open Source |
| `claude-sonnet-4-6` (or `claude-sonnet-4.6`) | `claude-sonnet-4-6` | Enabled | Claude |
| `claude-opus-4.6` (or `claude-opus-4-6-thinking`) | `claude-opus-4-6-thinking` | Enabled | Claude |
| `claude-sonnet-5-5-high` | `claude-sonnet-5-5-high` | High | Claude 5.5 |
| `claude-opus-5-5-high` | `claude-opus-5-5-high` | High | Claude 5.5 |

> **Dynamic Discovery:** Under `--backend agy-cli`, `GET /v1/models` queries `agy models` on demand to reflect all models supported by the user's active Antigravity account tier.

---

## Common Issues and Solutions

### 1. `agy` command not found
- Cause: The Antigravity CLI binary is not in system PATH.
- Fix:
  - Add `agy` to PATH, or
  - Specify the explicit binary path using `--agy-bin`:
    ```bash
    python antigravity_acp_bridge.py --agy-bin "C:\Users\<user>\AppData\Local\Programs\Antigravity\agy.exe"
    ```

### 2. "Token file not found" or "Preflight check failed"
- Cause: The user hasn't authenticated `agy` yet.
- Fix: Run `agy` in a normal terminal window and log in with their Google account.

### 3. "Verification Required" or Account Linking Notice
- Cause: Google temporarily flagged the account requiring standard browser confirmation.
- Behavior: The bridge stream-head inspector catches this immediately and outputs an HTTP 403 error containing the exact Google URL:
  `https://accounts.google.com/signin/...auth_success_gemini...`
- Fix: Tell the user to copy and open that URL in their logged-in browser to complete verification, then retry.

### 4. "Refusing to bind non-localhost host '0.0.0.0' without --api-key"
- Cause: Security feature preventing accidental public exposure of Google quota.
- Fix: Provide `--api-key "YOUR_SECRET"` (or `--allow-unauthenticated` if behind an isolated private network).

### 5. High concurrency or subprocess exhaustion
- Cause: Too many simultaneous requests spawning too many `agy` instances.
- Fix: The bridge features a built-in worker pool. Configure `--max-concurrency 1` or `--max-concurrency 2` to serialize requests safely without quota spikes.

---

## Architecture

```
[Default & Recommended: --backend agy-cli]
Client (OpenAI format, Vision, Streaming)
   │ HTTP POST /v1/chat/completions
   ▼
Bridge Server (BridgeHTTPServer / BridgeHandler)
   │
   ├─ Converts OpenAI message history & vision images into ACP ContentBlocks
   ├─ Manages warm ACP Worker Pool (agy acp --stdio) via JSON-RPC
   │
   ▼
Google Antigravity CLI (agy acp --stdio)
   │ Native Google Client Fingerprint & Telemetry
   ▼
Google AI Pro Upstream Backend
   │ Real-time token streaming
   ▼
Stream-head Inspection Guard
   ├─ Detects ToS / Account Verification banners before outputting
   └─ Extracts Google verification link if prompt is blocked
   ▼
OpenAI SSE Chunks or JSON Response back to Client
```

```
[Legacy / Fallback: --backend oauth]
Client (OpenAI format)
   │
   ▼
Bridge Translator (build_gemini_request)
   │
   ▼
Upstream HTTP Client (cloudcode-pa.googleapis.com)
   │ TokenManager refreshes OAuth token automatically
   ▼
Gemini to OpenAI Translator (gemini_response_to_openai)
```

---

## Project Structure Reference

| Path | Purpose |
|---|---|
| `antigravity_acp_bridge.py` | Top-level CLI launcher script |
| `antigravity_acp_bridge/` | Canonical Python package |
| `├── __init__.py` | Package version and exports |
| `├── __main__.py` | Module entrypoint (`python -m antigravity_acp_bridge`) |
| `├── cli.py` | Argument parsing, logging, and startup sequence |
| `├── config.py` | Configuration dataclass (`BridgeConfig`), constants, model mappings |
| `├── agy_cli.py` | ACP JSON-RPC client (`AgyAcpWorker`, `AgyCliClient`), worker pool, vision extractor, banner guard |
| `├── server.py` | `BridgeHTTPServer`, `BridgeHandler` (OpenAI routes, SSE, CORS, auth) |
| `├── translator.py` | OpenAI <-> Gemini request/response mapping (for OAuth mode) |
| `├── auth.py` | `TokenManager` & `ProjectResolver` (for OAuth mode) |
| `├── upstream.py` | Upstream HTTP client with retries (for OAuth mode) |
| `├── sse.py` | Server-Sent Events parser |
| `└── _http.py` | Pure stdlib HTTP request wrapper |
| `antigravity-acp-bridge.service` | Linux systemd user service unit template |
| `pyproject.toml` | Packaging metadata and `antigravity-acp-bridge` command |
| `tests/` | Complete test suite (38 unit & integration tests) |
| `README.md` | Human-facing documentation and usage guide |
| `AGENTS.md` | This file — reference instructions for AI assistants |

---

## Security Guidelines for AI Assistants

- **Localhost by default:** The bridge binds strictly to `127.0.0.1` unless explicitly requested.
- **API Key protection:** When instructing users to expose the bridge over LAN (`0.0.0.0`), always include `--api-key` in the command.
- **Zero token leakage:** Credentials and tokens stay exclusively on the local machine and communicate only through official Google endpoints.
