"""
snowflake_app/views.py

OpenAI-compatible endpoints that proxy to Snowflake Cortex.
Acts as a wrapper so any OpenAI SDK client can hit this Django app
and transparently use Cortex LLMs (mistral-large2, llama3.1-70b, etc.).

Cortex base URL is read from env: CORTEX_BASE_URL (default: http://192.168.16.113:8432/v1)
"""

import json
import os
import logging

import requests
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

_cortex_host = os.environ.get("CORTEX_HOST", "192.168.16.113")
_cortex_port = os.environ.get("CORTEX_PORT", "8432")
CORTEX_BASE_URL = os.environ.get("CORTEX_BASE_URL", f"http://{_cortex_host}:{_cortex_port}/v1")
CORTEX_API_KEY = os.environ.get("CORTEX_API_KEY", "cortex")

# Default model if none specified
DEFAULT_MODEL = os.environ.get("CORTEX_DEFAULT_MODEL", "mistral-large2")

SUPPORTED_MODELS = [
    "mistral-large2",
    "llama3.1-70b",
    "llama3.1-8b",
    "snowflake-arctic",
    "reka-flash",
    "reka-core",
]


def _cortex_headers():
    return {
        "Authorization": f"Bearer {CORTEX_API_KEY}",
        "Content-Type": "application/json",
    }


@csrf_exempt
@require_http_methods(["POST"])
def chat_completions(request):
    """
    POST /snowflake/v1/chat/completions
    OpenAI-compatible chat completions — proxied to Snowflake Cortex.
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    model = body.get("model", DEFAULT_MODEL)
    stream = body.get("stream", False)

    # Forward to Cortex
    cortex_url = f"{CORTEX_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "messages": body.get("messages", []),
        "temperature": body.get("temperature", 0.7),
        "max_tokens": body.get("max_tokens", 2048),
        "stream": stream,
    }

    # Pass through optional params if present
    for opt in ("top_p", "stop", "presence_penalty", "frequency_penalty"):
        if opt in body:
            payload[opt] = body[opt]

    try:
        if stream:
            def event_stream():
                with requests.post(
                    cortex_url,
                    json=payload,
                    headers=_cortex_headers(),
                    stream=True,
                    timeout=60,
                ) as resp:
                    for chunk in resp.iter_content(chunk_size=None):
                        yield chunk

            return StreamingHttpResponse(
                event_stream(),
                content_type="text/event-stream",
            )
        else:
            resp = requests.post(
                cortex_url,
                json=payload,
                headers=_cortex_headers(),
                timeout=60,
            )
            resp.raise_for_status()
            return JsonResponse(resp.json(), safe=False)

    except requests.exceptions.ConnectionError:
        logger.error("Cannot reach Cortex at %s", cortex_url)
        return JsonResponse(
            {"error": {"message": f"Cortex unreachable at {CORTEX_BASE_URL}", "type": "connection_error"}},
            status=503,
        )
    except requests.exceptions.HTTPError as e:
        logger.error("Cortex HTTP error: %s", e)
        return JsonResponse(
            {"error": {"message": str(e), "type": "cortex_error"}},
            status=resp.status_code,
        )


@csrf_exempt
@require_http_methods(["GET"])
def list_models(request):
    """
    GET /snowflake/v1/models
    Returns available Cortex models in OpenAI format.
    """
    models = [
        {
            "id": m,
            "object": "model",
            "owned_by": "snowflake-cortex",
            "permission": [],
        }
        for m in SUPPORTED_MODELS
    ]
    return JsonResponse({"object": "list", "data": models})


@csrf_exempt
@require_http_methods(["POST"])
def ask(request):
    """
    POST /snowflake/ask/
    Simple single-turn shorthand: { "prompt": "...", "model": "..." }
    Returns: { "response": "..." }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JsonResponse({"error": "prompt is required"}, status=400)

    model = body.get("model", DEFAULT_MODEL)

    cortex_url = f"{CORTEX_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": body.get("temperature", 0.7),
        "max_tokens": body.get("max_tokens", 1024),
    }

    try:
        resp = requests.post(
            cortex_url,
            json=payload,
            headers=_cortex_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        return JsonResponse({"response": reply, "model": model})
    except Exception as e:
        logger.error("Cortex /ask error: %s", e)
        return JsonResponse({"error": str(e)}, status=502)


@require_http_methods(["GET"])
def health(request):
    """GET /snowflake/health/ — quick liveness check"""
    return JsonResponse({"status": "ok", "cortex_base": CORTEX_BASE_URL})
