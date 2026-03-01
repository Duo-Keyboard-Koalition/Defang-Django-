"""
cursor_app/agent.py

LlamaIndex ReAct coding agent backed by Snowflake Cortex.
No external API keys — everything runs inside Snowflake.

Tools the agent has:
  - search_codebase(query)     → vector search over DKK.COMMUNITY.REPO_FILES
  - read_file(repo, path)      → fetch file content from Snowflake
  - list_repos()               → list indexed repos from GITHUB_REPOS
  - index_file(repo, path, content) → upsert file + embedding into REPO_FILES
"""

import os
import json
import hashlib
import snowflake.connector
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai_like import OpenAILike

# ─── Snowflake connection ─────────────────────────────────────────────────────

def _get_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database="DKK",
        schema="COMMUNITY",
    )


# ─── Tools ────────────────────────────────────────────────────────────────────

def search_codebase(query: str, repo_name: str = None, limit: int = 8) -> str:
    """
    Semantic search over all indexed code files in Snowflake.
    Returns file paths and relevant content snippets.

    Args:
        query: Natural language or code search query
        repo_name: Optional repo name to scope the search
        limit: Max results (default 8)
    """
    try:
        conn = _get_conn()
        cs = conn.cursor()
        # Use Cortex embedding for vector search
        if repo_name:
            cs.execute("""
                SELECT f.path, LEFT(f.content, 500) as snippet,
                       VECTOR_COSINE_SIMILARITY(
                           f.embedding,
                           SNOWFLAKE.CORTEX.EMBED_TEXT('e5-base-v2', %s)
                       ) as score
                FROM COMMUNITY.REPO_FILES f
                JOIN COMMUNITY.GITHUB_REPOS r ON f.repo_id = r.repo_id
                WHERE r.name = %s AND f.embedding IS NOT NULL
                ORDER BY score DESC LIMIT %s
            """, (query, repo_name, limit))
        else:
            cs.execute("""
                SELECT f.path, LEFT(f.content, 500) as snippet,
                       VECTOR_COSINE_SIMILARITY(
                           f.embedding,
                           SNOWFLAKE.CORTEX.EMBED_TEXT('e5-base-v2', %s)
                       ) as score
                FROM COMMUNITY.REPO_FILES f
                WHERE f.embedding IS NOT NULL
                ORDER BY score DESC LIMIT %s
            """, (query, limit))

        rows = cs.fetchall()
        conn.close()
        if not rows:
            return "No indexed files found. Files need to be indexed first with index_file()."
        results = []
        for path, snippet, score in rows:
            results.append(f"[{score:.3f}] {path}\n{snippet}")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Search error: {e}"


def read_file(repo_name: str, path: str) -> str:
    """
    Read full file content from Snowflake REPO_FILES index.

    Args:
        repo_name: Repository name (e.g. 'Defang-Django-Snowflake-OpenAI-Wrapper')
        path: File path within repo (e.g. 'app/cursor_app/agent.py')
    """
    try:
        conn = _get_conn()
        cs = conn.cursor()
        cs.execute("""
            SELECT f.content, f.language, f.updated_at
            FROM COMMUNITY.REPO_FILES f
            JOIN COMMUNITY.GITHUB_REPOS r ON f.repo_id = r.repo_id
            WHERE r.name = %s AND f.path = %s
        """, (repo_name, path))
        row = cs.fetchone()
        conn.close()
        if not row:
            return f"File not found in index: {path} in {repo_name}"
        content, lang, updated = row
        return f"# {path} ({lang}, indexed: {updated})\n\n{content}"
    except Exception as e:
        return f"Read error: {e}"


def list_repos() -> str:
    """List all repositories indexed in Snowflake with their stats."""
    try:
        conn = _get_conn()
        cs = conn.cursor()
        cs.execute("""
            SELECT r.name, r.language, r.stars, r.last_indexed,
                   COUNT(f.id) as file_count
            FROM COMMUNITY.GITHUB_REPOS r
            LEFT JOIN COMMUNITY.REPO_FILES f ON f.repo_id = r.repo_id
            GROUP BY r.name, r.language, r.stars, r.last_indexed
            ORDER BY file_count DESC
            LIMIT 20
        """)
        rows = cs.fetchall()
        conn.close()
        if not rows:
            return "No repos found."
        lines = ["Repo | Language | Stars | Files Indexed | Last Indexed"]
        lines.append("-" * 60)
        for name, lang, stars, last_indexed, count in rows:
            lines.append(f"{name} | {lang or 'N/A'} | {stars or 0} | {count} | {last_indexed or 'Never'}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def index_file(repo_name: str, path: str, content: str, language: str = None) -> str:
    """
    Index a file into Snowflake REPO_FILES with a Cortex embedding.
    Use this when working on code to keep the warehouse up to date.

    Args:
        repo_name: Repository name
        path: File path within repo
        content: File content to index
        language: Programming language (optional, auto-detected from extension)
    """
    try:
        conn = _get_conn()
        cs = conn.cursor()

        # Get repo_id
        cs.execute("SELECT repo_id FROM COMMUNITY.GITHUB_REPOS WHERE name = %s", (repo_name,))
        row = cs.fetchone()
        if not row:
            conn.close()
            return f"Repo '{repo_name}' not found in GITHUB_REPOS."
        repo_id = row[0]

        # Auto-detect language from extension
        if not language:
            ext_map = {
                '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
                '.tsx': 'TypeScript', '.jsx': 'JavaScript', '.go': 'Go',
                '.rs': 'Rust', '.java': 'Java', '.rb': 'Ruby',
                '.sql': 'SQL', '.md': 'Markdown', '.yaml': 'YAML', '.yml': 'YAML',
            }
            ext = '.' + path.rsplit('.', 1)[-1] if '.' in path else ''
            language = ext_map.get(ext, 'Text')

        file_id = f"{repo_id}:{hashlib.sha256(path.encode()).hexdigest()[:16]}"

        # Upsert file content
        cs.execute("""
            MERGE INTO COMMUNITY.REPO_FILES AS target
            USING (SELECT %s AS id, %s AS repo_id, %s AS path, %s AS content,
                          %s AS language, %s AS size_bytes) AS src
            ON target.id = src.id
            WHEN MATCHED THEN UPDATE SET
                content = src.content, language = src.language,
                size_bytes = src.size_bytes, updated_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT
                (id, repo_id, path, content, language, size_bytes)
                VALUES (src.id, src.repo_id, src.path, src.content, src.language, src.size_bytes)
        """, (file_id, repo_id, path, content, language, len(content)))

        # Generate and store embedding
        cs.execute("""
            UPDATE COMMUNITY.REPO_FILES
            SET embedding = SNOWFLAKE.CORTEX.EMBED_TEXT('e5-base-v2', %s)
            WHERE id = %s
        """, (content[:4000], file_id))  # truncate for embedding limits

        # Update last_indexed on repo
        cs.execute("""
            UPDATE COMMUNITY.GITHUB_REPOS SET last_indexed = CURRENT_TIMESTAMP()
            WHERE repo_id = %s
        """, (repo_id,))

        conn.close()
        return f"Indexed {path} ({language}, {len(content)} chars) into {repo_name} ✅"
    except Exception as e:
        return f"Index error: {e}"


# ─── LLM via Snowflake Cortex proxy ──────────────────────────────────────────

def build_agent(model: str = None) -> ReActAgent:
    """Build a ReAct agent using the local Cortex proxy as the LLM backend."""
    _model = model or os.environ.get("CORTEX_MODEL", "claude-opus-4-6")

    # Point LlamaIndex at our local Cortex proxy
    llm = OpenAILike(
        model=_model,
        api_base="http://localhost:8000/cortex/v1",
        api_key="dkk",
        is_chat_model=True,
        context_window=32000,
        max_tokens=4096,
    )

    tools = [
        FunctionTool.from_defaults(fn=search_codebase, name="search_codebase"),
        FunctionTool.from_defaults(fn=read_file, name="read_file"),
        FunctionTool.from_defaults(fn=list_repos, name="list_repos"),
        FunctionTool.from_defaults(fn=index_file, name="index_file"),
    ]

    return ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        max_iterations=10,
        context="""You are a coding assistant with access to a Snowflake-backed code warehouse.
You can search, read, and index code files. When asked about code, always search first.
When you write or modify code, index it back into Snowflake to keep the warehouse current.""",
    )
