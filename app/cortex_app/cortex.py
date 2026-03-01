"""
Cortex - Snowflake Cortex wrapper for the Defang Django stack.

Uses Snowflake CORTEX.COMPLETE() to run models natively — no OpenAI dependency.

Supported models (Snowflake Cortex):
  - snowflake-arctic          (Snowflake's own model)
  - mistral-large / mistral-7b
  - llama3.1-70b / llama3.1-8b / llama3-8b / llama3-70b
  - gemma-7b
  - reka-flash / reka-core
  - mixtral-8x7b
  - jamba-instruct / jamba-1.5-mini / jamba-1.5-large

Env vars:
  SNOWFLAKE_ACCOUNT    e.g. ymuajwd-ym41388
  SNOWFLAKE_USER       e.g. d273liu
  SNOWFLAKE_PASSWORD   (or use SNOWFLAKE_PRIVATE_KEY_PATH)
  SNOWFLAKE_WAREHOUSE  e.g. COMPUTE_WH
  SNOWFLAKE_DATABASE   e.g. AGENT_MEMORY
  SNOWFLAKE_SCHEMA     e.g. MAIN
  CORTEX_MODEL         default model name (default: snowflake-arctic)
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("CORTEX_MODEL", "claude-opus-4-5")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "CORTEX_SYSTEM_PROMPT",
    "You are Cortex, a helpful AI assistant embedded in the DKK platform."
)


def _get_connection():
    """Return a Snowflake connector connection."""
    import snowflake.connector
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ.get("SNOWFLAKE_PASSWORD", ""),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "AGENT_MEMORY"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "MAIN"),
        session_parameters={"QUERY_TAG": "cortex_app"},
    )


def chat(
    user_message: str,
    system_prompt: str = None,
    model: str = None,
    history: list = None,
    **kwargs
) -> dict:
    """
    Run a prompt through Snowflake CORTEX.COMPLETE().

    Args:
        user_message: The user's message.
        system_prompt: Optional system prompt override.
        model: Snowflake Cortex model name (default: CORTEX_MODEL env / snowflake-arctic).
        history: Optional list of prior messages [{"role": ..., "content": ...}].

    Returns:
        dict with keys: content, model, usage (approximate), finish_reason
    """
    _model = model or DEFAULT_MODEL
    _system = system_prompt or DEFAULT_SYSTEM_PROMPT

    # Build message array for CORTEX.COMPLETE multi-turn format
    messages = [{"role": "system", "content": _system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    conn = _get_connection()
    try:
        cur = conn.cursor()
        # Build ARRAY_CONSTRUCT with OBJECT_CONSTRUCT for each message
        obj_parts = []
        for m in messages:
            role = m["role"].replace("'", "\\'")
            content = m["content"].replace("'", "\\'")
            obj_parts.append(f"OBJECT_CONSTRUCT('role', '{role}', 'content', '{content}')")
        array_sql = f"ARRAY_CONSTRUCT({', '.join(obj_parts)})"
        cur.execute(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{_model}', {array_sql}) AS response")
        row = cur.fetchone()
        if not row or not row[0]:
            raise ValueError("Empty response from Snowflake Cortex")

        result = json.loads(row[0]) if isinstance(row[0], str) else row[0]

        # Snowflake Cortex response shape
        choices = result.get("choices", [{}])
        content = choices[0].get("messages", "") if choices else ""
        usage = result.get("usage", {})

        return {
            "content": content,
            "model": _model,
            "usage": usage,
            "finish_reason": "stop",
        }
    finally:
        conn.close()


def list_models() -> list:
    """Return the list of models available in Snowflake Cortex."""
    return [
        # Anthropic Claude (private preview, US — Feb 2026)
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
        # AI21 Jamba
        "jamba-instruct",
        "jamba-1.5-mini",
        "jamba-1.5-large",
    ]
