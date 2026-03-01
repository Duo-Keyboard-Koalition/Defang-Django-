"""
Cortex - OpenAI wrapper/gateway for the Defang Django stack.

Provides a unified interface to OpenAI completions that can be
called by other apps within this Django project or via the REST API.
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("CORTEX_MODEL", "gpt-4o")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "CORTEX_SYSTEM_PROMPT",
    "You are Cortex, a helpful AI assistant embedded in the DKK platform."
)


def chat(
    user_message: str,
    system_prompt: str = None,
    model: str = None,
    history: list = None,
    **kwargs
) -> dict:
    """
    Send a message to OpenAI and return the response.

    Args:
        user_message: The user's message.
        system_prompt: Optional override for the system prompt.
        model: Optional model override (default: CORTEX_MODEL env var or gpt-4o).
        history: Optional list of prior messages [{"role": ..., "content": ...}].
        **kwargs: Extra params forwarded to openai.ChatCompletion.create().

    Returns:
        dict with keys: content, model, usage, finish_reason
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        # Fallback for openai<1.0
        import openai as _openai
        _openai.api_key = OPENAI_API_KEY
        client = None

    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    _model = model or DEFAULT_MODEL

    if client:
        response = client.chat.completions.create(
            model=_model,
            messages=messages,
            **kwargs
        )
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage),
            "finish_reason": response.choices[0].finish_reason,
        }
    else:
        import openai as _openai
        response = _openai.ChatCompletion.create(
            model=_model,
            messages=messages,
            **kwargs
        )
        return {
            "content": response["choices"][0]["message"]["content"],
            "model": response["model"],
            "usage": dict(response["usage"]),
            "finish_reason": response["choices"][0]["finish_reason"],
        }
