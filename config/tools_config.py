"""Tools configuration – environment-driven settings for MCP tools."""

from __future__ import annotations

import os


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


TOOLS_CONFIG = {
    # ── Safety ──────────────────────────────────────────────────────────
    # When True, tools marked requires_approval=True will be blocked until
    # the user explicitly confirms the action.
    'APPROVAL_MODE': os.getenv('APPROVAL_MODE', 'false').lower() == 'true',

    # Maximum seconds a single tool call may run before being abandoned.
    'TOOL_TIMEOUT': _safe_int(os.getenv('TOOL_TIMEOUT'), default=30),

    # Maximum number of tool-call rounds per user request (prevents loops).
    'MAX_TOOL_ITERATIONS': _safe_int(os.getenv('MAX_TOOL_ITERATIONS'), default=5),

    # ── MCP HTTP Server (optional) ───────────────────────────────────────
    # Set MCP_SERVER_ENABLED=true to expose tools over HTTP/SSE.
    'MCP_SERVER_ENABLED': os.getenv('MCP_SERVER_ENABLED', 'false').lower() == 'true',
    'MCP_SERVER_HOST': os.getenv('MCP_SERVER_HOST', '127.0.0.1'),
    'MCP_SERVER_PORT': _safe_int(os.getenv('MCP_SERVER_PORT'), default=8765),

    # ── Session Deduplication ────────────────────────────────────────────
    # Seconds after a session completes during which a bare confirmation
    # message is suppressed (not treated as a new task trigger).
    # Set to 0 to disable deduplication.
    'SESSION_DEDUP_WINDOW_SECONDS': _safe_int(os.getenv('SESSION_DEDUP_WINDOW_SECONDS'), default=120),

    # ── Knowledge Base ───────────────────────────────────────────────────
    'KNOWLEDGE_STORE_FILE': os.getenv('KNOWLEDGE_STORE_FILE', 'jarvis_knowledge.json'),

    # ── Web APIs ─────────────────────────────────────────────────────────
    # Free tier – register at https://openweathermap.org/api
    'OPENWEATHER_API_KEY': os.getenv('OPENWEATHER_API_KEY', ''),

    # ── Home Automation ──────────────────────────────────────────────────
    # Base URL of your smart-home hub (e.g. Home Assistant: http://homeassistant.local:8123)
    'HOME_AUTOMATION_URL': os.getenv('HOME_AUTOMATION_URL', ''),
}
