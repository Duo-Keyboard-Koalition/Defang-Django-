"""
cursor_app/views.py

OpenAI-compatible endpoint for Cursor backed by a LlamaIndex ReAct agent.
LLM: Snowflake Cortex (no external API keys).
Storage: Snowflake DKK.COMMUNITY.REPO_FILES (vector search + code indexing).

Cursor config:
  Base URL: http://192.168.16.124:8000/cursor/v1
  API Key:  dkk (anything)
  Model:    claude-opus-4-6
"""

import json
import os
import time
import uuid
import logging

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-5",
    "mistral-large2",
    "snowflake-arctic",
    "llama3.1-70b",
]


@csrf_exempt
def chat_completions(request):
    if request.method == "OPTIONS":
        return _cors(JsonResponse({}))
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return _cors(JsonResponse({"error": "Invalid JSON"}, status=400))

    messages = body.get("messages", [])
    model = body.get("model", os.environ.get("CORTEX_MODEL", "claude-opus-4-6"))

    # Extract last user message for the agent
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        return _cors(JsonResponse({
            "error": {"message": "No user message found", "type": "invalid_request_error"}
        }, status=400))

    try:
        from .agent import build_agent
        agent = build_agent(model=model)
        response = agent.chat(user_message)
        content = str(response)
    except Exception as exc:
        logger.exception("Agent error")
        # Fallback to plain Cortex if agent fails
        try:
            from cortex_app.cortex import chat
            result = chat(user_message, model=model)
            content = result.get("content", str(exc))
        except Exception:
            content = f"Agent error: {exc}"

    return _cors(JsonResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }))


@require_http_methods(["GET"])
def list_models(request):
    return _cors(JsonResponse({
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "snowflake-cortex"} for m in SUPPORTED_MODELS],
    }))


@require_http_methods(["GET"])
def health(request):
    return _cors(JsonResponse({
        "status": "ok",
        "service": "cursor_app",
        "backend": "llama-index-react-agent + snowflake-cortex",
        "tool_calling": True,
        "storage": "DKK.COMMUNITY.REPO_FILES",
        "default_model": os.environ.get("CORTEX_MODEL", "claude-opus-4-6"),
    }))


def _cors(r):
    r["Access-Control-Allow-Origin"] = "*"
    r["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    r["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return r
