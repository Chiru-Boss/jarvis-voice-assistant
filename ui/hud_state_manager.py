"""HUD state transition tracker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List


@dataclass(frozen=True)
class HUDStateEvent:
    timestamp: str
    state: str
    detail: str


class HUDStateManager:
    VALID_STATES = {'idle', 'listening', 'thinking', 'executing', 'speaking', 'success', 'error'}
    MAX_EVENT_HISTORY = 50

    def __init__(self):
        self._events: List[HUDStateEvent] = []
        self._state = 'idle'

    def transition(self, state: str, detail: str = '') -> None:
        state = state.lower().strip()
        if state not in self.VALID_STATES:
            raise ValueError(f'Unsupported HUD state: {state}')
        self._state = state
        self._events.append(HUDStateEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            state=state,
            detail=detail,
        ))

    def snapshot(self) -> Dict[str, object]:
        return {
            'state': self._state,
            'events': [asdict(e) for e in self._events[-self.MAX_EVENT_HISTORY:]],
        }
