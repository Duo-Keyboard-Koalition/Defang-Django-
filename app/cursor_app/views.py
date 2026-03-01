"""
Cursor integration layer — OpenAI-compat endpoint with Snowflake RAG.

Flow:
  Cursor → POST /cursor/v1/chat/completions
         → retrieve Snowflake context for the last user message
         → inject context into system prompt
         → forward to /snowflake/v1/chat/completions (Cortex)
         → stream response back to Cursor

Cursor config:
  Base URL: http://<host>:8000/cursor/v1
  API Key:  any non-empty string
  Model:    claude-opus-4-6 (or any Cortex model)
"""

import json
import logging
import os
import requests

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .snowflake_rag import retrieve_context

logger = logging.getLogger(__name__)

# Internal Snowflake endpoint (same Django server)
SNOWFLAKE_COMPLETIONS_URL = os.environ.get(
    'SNOWFLAKE_COMPLETIONS_URL',
    'http://localhost:8000/snowflake/v1/chat/completions'
)

AVAILABLE_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-5",
    "mistral-large2",
    "llama3.1-70b",
    "snowflake-arctic",
]

RAG_ENABLED = os.environ.get('CURSOR_RAG_ENABLED', 'true').lower() == 'true'


@csrf_exempt
@require_http_methods(["GET"])
def models(request):
    """GET /cursor/v1/models — list available models for Cursor."""
    return JsonResponse({
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "snowflake-cortex"}
            for m in AVAILABLE_MODELS
        ]
    })


@csrf_exempt
@require_http_methods(["POST"])
def chat_completions(request):
    """POST /cursor/v1/chat/completions — RAG-augmented Cortex proxy."""
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Inject Snowflake context into system prompt
    if RAG_ENABLED and messages:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            None
        )
        if last_user:
            context = retrieve_context(last_user)
            if context:
                # Prepend or augment the system message
                system_msgs = [m for m in messages if m.get("role") == "system"]
                if system_msgs:
                    system_msgs[0]["content"] = context + "\n\n" + system_msgs[0]["content"]
                else:
                    messages = [{"role": "system", "content": context}] + messages
                body["messages"] = messages
                logger.debug(f"Injected {len(context)} chars of Snowflake context")

    # Forward to Snowflake/Cortex endpoint
    try:
        resp = requests.post(
            SNOWFLAKE_COMPLETIONS_URL,
            json=body,
            stream=stream,
            timeout=120,
            headers={"Content-Type": "application/json"},
        )

        if stream:
            def event_stream():
                for chunk in resp.iter_content(chunk_size=None):
                    yield chunk

            return StreamingHttpResponse(
                event_stream(),
                content_type=resp.headers.get("Content-Type", "text/event-stream"),
                status=resp.status_code,
            )

        return JsonResponse(resp.json(), status=resp.status_code, safe=False)

    except requests.RequestException as e:
        logger.error(f"Upstream Cortex request failed: {e}")
        return JsonResponse({"error": str(e)}, status=502)


@csrf_exempt
@require_http_methods(["GET"])
def health(request):
    return JsonResponse({"status": "ok", "rag_enabled": RAG_ENABLED})
