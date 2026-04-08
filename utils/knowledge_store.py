"""Knowledge Store – simple keyword-based JSON-backed document storage."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List


class KnowledgeStore:
    """Persistent knowledge base backed by a local JSON file.

    Entries are stored as a list of dicts with ``title``, ``content``,
    ``tags``, and ``timestamp`` fields.  Retrieval uses simple keyword
    matching (each query word scores +1 for each field it appears in).
    """

    def __init__(self, store_file: str = 'jarvis_knowledge.json') -> None:
        self.store_file = store_file
        self._entries: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load entries from disk; silently start empty on any error."""
        if not os.path.exists(self.store_file):
            return
        try:
            with open(self.store_file, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            self._entries = data.get('entries', [])
        except (json.JSONDecodeError, IOError):
            self._entries = []

    def _save(self) -> None:
        """Persist entries to disk."""
        try:
            with open(self.store_file, 'w', encoding='utf-8') as fh:
                json.dump({'entries': self._entries}, fh, indent=2, ensure_ascii=False)
        except IOError as exc:
            print(f"⚠️  Could not save knowledge store: {exc}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, title: str, content: str, tags: List[str] | None = None) -> None:
        """Append a new entry and persist."""
        entry: Dict[str, Any] = {
            'id': len(self._entries) + 1,
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'content': content,
            'tags': tags or [],
        }
        self._entries.append(entry)
        self._save()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Return up to *max_results* entries that best match *query*.

        Scoring: each query keyword that appears in title/content/tags
        contributes +1 to the entry's score.  Entries with score 0 are
        excluded.
        """
        keywords = query.lower().split()
        scored: List[tuple[int, Dict[str, Any]]] = []

        for entry in self._entries:
            haystack = (
                f"{entry['title']} {entry['content']} "
                f"{' '.join(entry.get('tags', []))}"
            ).lower()
            score = sum(1 for kw in keywords if kw in haystack)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                'title': e['title'],
                'content': e['content'][:500],
                'timestamp': e['timestamp'],
                'tags': e.get('tags', []),
            }
            for _, e in scored[:max_results]
        ]

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the *n* most recently added entries."""
        return [
            {
                'title': e['title'],
                'content': e['content'][:500],
                'timestamp': e['timestamp'],
                'tags': e.get('tags', []),
            }
            for e in self._entries[-n:]
        ]

    def summary(self) -> str:
        count = len(self._entries)
        return f"{count} entr{'ies' if count != 1 else 'y'} in knowledge base"
