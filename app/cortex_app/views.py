"""
Cortex REST API views — OpenAI wrapper endpoints.

Endpoints:
  POST /cortex/chat/        — single-turn or multi-turn chat
  GET  /cortex/health/      — liveness check
"""

import json
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from .cortex import chat, DEFAULT_MODEL


@csrf_exempt
@require_http_methods(["POST"])
def cortex_chat(request):
    """
    OpenAI chat completion endpoint.

    Request body (JSON):
      {
        "message": "Your question here",           // required
        "system_prompt": "Optional override",      // optional
        "model": "gpt-4o",                         // optional
        "history": [                               // optional
          {"role": "user",      "content": "..."},
          {"role": "assistant", "content": "..."}
        ]
      }

    Response:
      {
        "content": "...",
        "model": "gpt-4o",
        "usage": {...},
        "finish_reason": "stop"
      }
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    user_message = body.get("message", "").strip()
    if not user_message:
        return JsonResponse({"error": "'message' field is required"}, status=400)

    system_prompt = body.get("system_prompt")
    model = body.get("model")
    history = body.get("history", [])

    # Basic history validation
    if not isinstance(history, list):
        return JsonResponse({"error": "'history' must be a list"}, status=400)

    try:
        result = chat(
            user_message=user_message,
            system_prompt=system_prompt,
            model=model,
            history=history,
        )
        return JsonResponse(result)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@require_http_methods(["GET"])
def cortex_health(request):
    """Liveness / readiness check."""
    return JsonResponse({
        "status": "ok",
        "service": "cortex",
        "default_model": DEFAULT_MODEL,
    })
