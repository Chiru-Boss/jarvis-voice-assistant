from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List


@dataclass
class HUDState:
    state: str
    message: str
    timestamp: str


class HUDStateManager:
    """Stores real-time HUD state transitions for overlay/tray consumers."""

    def __init__(self, max_events: int = 100) -> None:
        self._events: Deque[HUDState] = deque(maxlen=max_events)
        self._current = HUDState(state='idle', message='Ready', timestamp=self._now())

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def set_state(self, state: str, message: str = '') -> None:
        event = HUDState(state=state, message=message, timestamp=self._now())
        self._current = event
        self._events.append(event)

    def snapshot(self) -> Dict[str, str]:
        return {
            'state': self._current.state,
            'message': self._current.message,
            'timestamp': self._current.timestamp,
        }

    def events(self) -> List[Dict[str, str]]:
        return [{'state': e.state, 'message': e.message, 'timestamp': e.timestamp} for e in self._events]
