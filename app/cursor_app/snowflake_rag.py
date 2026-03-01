"""
Snowflake RAG retrieval for Cursor context injection.

Queries repo/file embeddings from Snowflake and returns relevant context
to inject into the system prompt before forwarding to Cortex.

Schema expected (GalClaw's design):
  DARK_FORGE.DKK.REPO_FILES      (repo_name, file_path, content, embedding VECTOR)
  DARK_FORGE.DKK.REPO_COMMITS    (repo_name, sha, message, author, timestamp)

Falls back to keyword search if embeddings table isn't ready yet.
"""

import os
import logging
import snowflake.connector

logger = logging.getLogger(__name__)

SNOWFLAKE_CONFIG = {
    'user': os.environ.get('SNOWFLAKE_USER', 'd273liu'),
    'password': os.environ.get('SNOWFLAKE_PASSWORD', ''),
    'account': os.environ.get('SNOWFLAKE_ACCOUNT', 'ymuajwd-ym41388'),
    'warehouse': os.environ.get('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
    'database': os.environ.get('SNOWFLAKE_DATABASE', 'DARK_FORGE'),
    'schema': os.environ.get('SNOWFLAKE_SCHEMA', 'DKK'),
}

MAX_CONTEXT_CHARS = 4000
MAX_FILES = 5


def _get_conn():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)


def retrieve_context(query: str, repo_filter: str = None) -> str:
    """
    Retrieve relevant repo/file context for a query.
    Returns a formatted string ready for system prompt injection.
    Falls back gracefully if tables don't exist yet.
    """
    try:
        return _vector_search(query, repo_filter)
    except Exception as e:
        logger.warning(f"Vector search unavailable ({e}), trying keyword fallback")
        try:
            return _keyword_search(query, repo_filter)
        except Exception as e2:
            logger.warning(f"Keyword search also failed ({e2}), returning empty context")
            return ""


def _vector_search(query: str, repo_filter: str = None) -> str:
    """Use Cortex EMBED_TEXT + VECTOR_COSINE_SIMILARITY for semantic search."""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        repo_clause = f"AND repo_name = '{repo_filter}'" if repo_filter else ""
        # Use Cortex to embed the query inline
        sql = f"""
            SELECT file_path, content, repo_name,
                   VECTOR_COSINE_SIMILARITY(
                       embedding,
                       SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', %s)
                   ) AS score
            FROM DARK_FORGE.DKK.REPO_FILES
            WHERE 1=1 {repo_clause}
            ORDER BY score DESC
            LIMIT {MAX_FILES}
        """
        cur.execute(sql, (query,))
        rows = cur.fetchall()
        return _format_context(rows)
    finally:
        conn.close()


def _keyword_search(query: str, repo_filter: str = None) -> str:
    """Simple ILIKE keyword fallback when embeddings aren't ready."""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        repo_clause = f"AND repo_name = '{repo_filter}'" if repo_filter else ""
        keywords = ' OR '.join([f"content ILIKE '%{w}%'" for w in query.split()[:5]])
        sql = f"""
            SELECT file_path, content, repo_name, 0.5 AS score
            FROM DARK_FORGE.DKK.REPO_FILES
            WHERE ({keywords}) {repo_clause}
            LIMIT {MAX_FILES}
        """
        cur.execute(sql)
        rows = cur.fetchall()
        return _format_context(rows)
    finally:
        conn.close()


def _format_context(rows) -> str:
    if not rows:
        return ""
    parts = ["### Relevant repository context:\n"]
    total = 0
    for file_path, content, repo_name, score in rows:
        snippet = content[:800] if content else ""
        entry = f"**{repo_name}/{file_path}** (relevance: {score:.2f}):\n```\n{snippet}\n```\n"
        total += len(entry)
        if total > MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
    return "\n".join(parts)
