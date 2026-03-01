"""
Cortex REST API views — fully OpenAI-compatible.

Endpoints:
  POST /cortex/v1/chat/completions  — OpenAI-compat (streaming, tool calling)
  GET  /cortex/v1/models            — model list
  POST /cortex/chat/                — simple shorthand (non-streaming)
  GET  /cortex/health/              — liveness

Cursor / OpenAI SDK config:
  Base URL:  http://<host>/cortex/v1
  API Key:   anything (put "dkk")
  Model:     claude-opus-4-6
"""

import json
import logging

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .cortex import chat, list_models, DEFAULT_MODEL, SNOWFLAKE_ACCOUNT

logger = logging.getLogger(__name__)


# ─── OpenAI-compatible endpoints ────────────────────────────────────────────

@csrf_exempt
def chat_completions(request):
    """
    POST /cortex/v1/chat/completions

    Fully OpenAI-compatible. Supports:
      - streaming (stream: true) — returns SSE
      - tool calling (tools / tool_choice)
      - multi-turn history (messages array)

    Use this base URL in Cursor: http://<host>/cortex/v1
    """
    if request.method not in ("POST", "OPTIONS"):
        return JsonResponse({"error": "Method not allowed"}, status=405)
    if request.method == "OPTIONS":
        return _cors(JsonResponse({}))

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return _cors(JsonResponse({"error": "Invalid JSON"}, status=400))

    messages = body.get("messages", [])
    model    = body.get("model", DEFAULT_MODEL)
    stream   = body.get("stream", False)
    tools    = body.get("tools")
    tool_choice = body.get("tool_choice")

    # Extract system + history from messages array
    system_prompt = None
    history = []
    user_message = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_message = content
            # don't break — keep last user message
        else:
            history.append(msg)

    if not user_message and messages:
        # fall back: treat last message content as user message
        user_message = messages[-1].get("content", "")

    try:
        result = chat(
            user_message=user_message,
            system_prompt=system_prompt,
            model=model,
            history=history,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
        )
    except Exception as exc:
        logger.exception("Cortex call failed")
        return _cors(JsonResponse({
            "error": {"message": str(exc), "type": "api_error"}
        }, status=500))

    if stream:
        # result is a requests.Response — forward SSE
        def _gen():
            try:
                for chunk in result.iter_lines():
                    if chunk:
                        yield chunk.decode() + "\n\n"
            finally:
                result.close()

        resp = StreamingHttpResponse(_gen(), content_type="text/event-stream")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        _add_cors(resp)
        return resp

    return _cors(JsonResponse(result))


@csrf_exempt
def models_list(request):
    """GET /cortex/v1/models — OpenAI-compat model list."""
    models = list_models()
    return _cors(JsonResponse({
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "snowflake-cortex"} for m in models],
    }))


# ─── Simple shorthand ────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def cortex_chat(request):
    """
    POST /cortex/chat/
    Simple non-streaming JSON shorthand (for curl testing / frontend_app).

    Body: {"message": "...", "model": "...", "system_prompt": "...", "history": [...]}
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    user_message = body.get("message", "").strip()
    if not user_message:
        return JsonResponse({"error": "'message' is required"}, status=400)

    try:
        result = chat(
            user_message=user_message,
            system_prompt=body.get("system_prompt"),
            model=body.get("model"),
            history=body.get("history", []),
        )
        return JsonResponse({
            "content": result.get("content", ""),
            "model": result.get("model", DEFAULT_MODEL),
            "usage": result.get("usage", {}),
            "finish_reason": result.get("finish_reason", "stop"),
        })
    except Exception as exc:
        logger.exception("cortex_chat failed")
        return JsonResponse({"error": str(exc)}, status=500)


@require_http_methods(["GET"])
def cortex_health(request):
    """GET /cortex/health/"""
    return _cors(JsonResponse({
        "status": "ok",
        "service": "cortex",
        "backend": "snowflake-cortex-rest",
        "account": SNOWFLAKE_ACCOUNT,
        "default_model": DEFAULT_MODEL,
        "tool_calling": True,
        "streaming": True,
    }))


# ─── CORS helpers ────────────────────────────────────────────────────────────

def _add_cors(response):
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

def _cors(response):
    return _add_cors(response)
