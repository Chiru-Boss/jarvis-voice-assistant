"""Small in-memory TTL cache for expensive vision calls."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional, Tuple


class VisionCache:
    def __init__(self, ttl_seconds: float = 60.0):
        self._ttl = max(1.0, float(ttl_seconds))
        self._items: Dict[str, Tuple[float, Any]] = {}

    @staticmethod
    def build_key(task: str, image_bytes: bytes) -> str:
        h = hashlib.sha256()
        h.update(task.encode('utf-8'))
        h.update(image_bytes)
        return h.hexdigest()

    def get(self, key: str) -> Optional[Any]:
        entry = self._items.get(key)
        if not entry:
            return None
        created_at, value = entry
        if (time.time() - created_at) > self._ttl:
            self._items.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = (time.time(), value)
