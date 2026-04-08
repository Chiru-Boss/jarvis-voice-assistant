"""Knowledge Base Tools – search, add, and retrieve from the personal knowledge store."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.tool_registry import ToolRegistry

# Module-level singleton so the store is shared across calls without
# requiring the caller to manage its lifetime.
_store: Optional[Any] = None


def _get_store():
    """Return the shared KnowledgeStore instance, creating it on first call."""
    global _store  # noqa: PLW0603
    if _store is None:
        from utils.knowledge_store import KnowledgeStore
        import os
        store_file = os.getenv('KNOWLEDGE_STORE_FILE', 'jarvis_knowledge.json')
        _store = KnowledgeStore(store_file=store_file)
    return _store


# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------

def search_knowledge_base(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Search the personal knowledge base for entries matching *query*."""
    return _get_store().search(query, max_results=max_results)


def add_to_knowledge(title: str, content: str, tags: str = '') -> str:
    """Add a new entry to the personal knowledge base."""
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    _get_store().add(title=title, content=content, tags=tag_list)
    return f"✅ Added '{title}' to the knowledge base."


def get_context(n: int = 5) -> List[Dict[str, Any]]:
    """Retrieve the *n* most recent entries from the knowledge base."""
    return _get_store().get_recent(n=n)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def register_tools(registry: ToolRegistry) -> None:
    """Register all knowledge base tools with *registry*."""

    registry.register(
        name='search_knowledge_base',
        description=(
            'Search your personal knowledge base for information from previous '
            'conversations and saved documents. Use this to recall what was '
            'discussed earlier or to look up saved notes.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Search query to match against stored entries.',
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of results to return (default 3).',
                },
            },
            'required': ['query'],
        },
        func=search_knowledge_base,
        safe=True,
    )

    registry.register(
        name='add_to_knowledge',
        description=(
            'Save new information to your personal knowledge base for future retrieval. '
            'Useful for storing facts, notes, or conversation summaries.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'title': {
                    'type': 'string',
                    'description': 'Short title or topic label for the entry.',
                },
                'content': {
                    'type': 'string',
                    'description': 'Full text content to store.',
                },
                'tags': {
                    'type': 'string',
                    'description': 'Comma-separated tags for easier retrieval.',
                },
            },
            'required': ['title', 'content'],
        },
        func=add_to_knowledge,
        safe=True,
    )

    registry.register(
        name='get_context',
        description=(
            'Retrieve the most recent entries from the knowledge base to get '
            'conversation context or recently saved information.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'n': {
                    'type': 'integer',
                    'description': 'Number of recent entries to retrieve (default 5).',
                },
            },
        },
        func=get_context,
        safe=True,
    )
