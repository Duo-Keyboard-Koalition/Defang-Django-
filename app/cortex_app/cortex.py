"""
Cortex - Snowflake Cortex REST API wrapper.

Uses the Snowflake Cortex REST API (not SQL CORTEX.COMPLETE) which gives us:
  - Full OpenAI-compatible response format
  - Streaming support (SSE)
  - Tool calling / function calling passthrough
  - All models: claude-opus-4-6, mistral-large2, llama3.1-70b, etc.

REST endpoint:
  https://<account>.snowflakecomputing.com/api/v2/cortex/inference:complete

Auth: Snowflake JWT (key-pair) or username/password token exchange.
Env vars:
  SNOWFLAKE_ACCOUNT    e.g. ymuajwd-ym41388
  SNOWFLAKE_USER       e.g. d273liu
  SNOWFLAKE_PASSWORD   (plaintext, used for token exchange)
  SNOWFLAKE_WAREHOUSE  e.g. COMPUTE_WH
  CORTEX_MODEL         default model (default: claude-opus-4-6)
"""

import os
import json
import time
import threading
import requests
from dotenv import load_dotenv

load_dotenv()

SNOWFLAKE_ACCOUNT  = os.getenv("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_USER     = os.getenv("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
DEFAULT_MODEL      = os.getenv("CORTEX_MODEL", "claude-opus-4-6")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "CORTEX_SYSTEM_PROMPT",
    "You are Cortex, a helpful AI assistant embedded in the DKK platform."
)

# Token cache: {token, expires_at}
_token_cache: dict = {}
_token_lock = threading.Lock()

SUPPORTED_MODELS = [
    # Anthropic Claude (Cortex private preview, US — Feb 2026)
    "claude-opus-4-6",
    "claude-opus-4-5",
    # Snowflake
    "snowflake-arctic",
    # Mistral
    "mistral-large2",
    "mistral-large",
    "mistral-7b",
    "mixtral-8x7b",
    # Meta Llama
    "llama3.1-70b",
    "llama3.1-8b",
    "llama3-70b",
    "llama3-8b",
    # Google
    "gemma-7b",
    # Reka
    "reka-flash",
    "reka-core",
    # AI21
    "jamba-instruct",
    "jamba-1.5-mini",
    "jamba-1.5-large",
]


def _snowflake_host() -> str:
    """Return the Snowflake host URL."""
    account = SNOWFLAKE_ACCOUNT.replace("_", "-").lower()
    return f"https://{account}.snowflakecomputing.com"


def _get_token() -> str:
    """
    Get a Snowflake session token via username/password.
    Caches the token until ~1 min before expiry.
    """
    with _token_lock:
        now = time.time()
        if _token_cache.get("token") and _token_cache.get("expires_at", 0) > now + 60:
            return _token_cache["token"]

        host = _snowflake_host()
        resp = requests.post(
            f"{host}/session/v1/login-request",
            params={"warehouse": SNOWFLAKE_WAREHOUSE},
            json={
                "data": {
                    "CLIENT_APP_ID": "DKK-Cortex",
                    "CLIENT_APP_VERSION": "1.0",
                    "SVN_REVISION": "1",
                    "ACCOUNT_NAME": SNOWFLAKE_ACCOUNT,
                    "LOGIN_NAME": SNOWFLAKE_USER,
                    "PASSWORD": SNOWFLAKE_PASSWORD,
                }
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data["data"]["token"]
        # Snowflake session tokens expire after ~4h by default
        _token_cache["token"] = token
        _token_cache["expires_at"] = now + 3600 * 3.5

        return token


def _cortex_url() -> str:
    return f"{_snowflake_host()}/api/v2/cortex/inference:complete"


def _build_headers() -> dict:
    token = _get_token()
    return {
        "Authorization": f'Snowflake Token="{token}"',
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Snowflake-Authorization-Token-Type": "OAUTH",
    }


def chat(
    user_message: str,
    system_prompt: str = None,
    model: str = None,
    history: list = None,
    tools: list = None,
    tool_choice=None,
    stream: bool = False,
    **kwargs,
) -> dict:
    """
    Call Snowflake Cortex REST API for a chat completion.

    Supports tool calling — pass tools/tool_choice to enable function calling
    compatible with OpenAI format (Cursor, etc. will work).

    Returns OpenAI-compatible response dict.
    """
    _model = model or DEFAULT_MODEL
    _system = system_prompt or DEFAULT_SYSTEM_PROMPT

    messages = [{"role": "system", "content": _system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": _model,
        "messages": messages,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    resp = requests.post(
        _cortex_url(),
        headers=_build_headers(),
        json=payload,
        timeout=120,
        stream=stream,
    )
    resp.raise_for_status()

    if stream:
        return resp  # caller handles streaming

    data = resp.json()
    # Cortex REST returns OpenAI-shaped response already
    choices = data.get("choices", [{}])
    msg = choices[0].get("message", {}) if choices else {}
    return {
        "id": data.get("id", ""),
        "object": "chat.completion",
        "model": data.get("model", _model),
        "choices": choices,
        "usage": data.get("usage", {}),
        # Convenience shortcut for non-OpenAI callers
        "content": msg.get("content", ""),
        "finish_reason": choices[0].get("finish_reason", "stop") if choices else "stop",
    }


def list_models() -> list:
    return SUPPORTED_MODELS
