"""UI bridge stubs for integrating external HUD/front-end systems.

This module provides a non-breaking event stream that can be consumed by a
future Whisperflowactions-style HUD without changing the core JARVIS pipeline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


@dataclass(frozen=True)
class UIEvent:
    """Single UI-facing event emitted by the voice pipeline."""

    timestamp: str
    state: str
    detail: str
    source: str
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None


class UIBridge:
    """Buffered event bridge between JARVIS core and external UIs."""

    def __init__(self, *, enabled: bool = False, max_events: int = 200):
        if max_events < 1:
            raise ValueError(f'max_events must be >= 1, got {max_events}')
        self.enabled = enabled
        self.max_events = int(max_events)
        self._events: Deque[UIEvent] = deque(maxlen=self.max_events)
        self._latest = UIEvent(
            timestamp=self._now(),
            state='IDLE',
            detail='JARVIS ready',
            source='bootstrap',
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def transition(self, state: str, detail: str = '', *, source: str = 'main', metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit a state transition for UI subscribers."""
        if not self.enabled:
            return
        event = UIEvent(
            timestamp=self._now(),
            state=state.upper(),
            detail=detail,
            source=source,
            stream=False,
            metadata=metadata,
        )
        self._latest = event
        self._events.append(event)

    def stream_update(self, detail: str, *, source: str = 'main', metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit an incremental stream-like update (token/chunk/progress)."""
        if not self.enabled:
            return
        event = UIEvent(
            timestamp=self._now(),
            state='STREAM',
            detail=detail,
            source=source,
            stream=True,
            metadata=metadata,
        )
        self._latest = event
        self._events.append(event)

    def snapshot(self) -> Dict[str, Any]:
        """Return current bridge state and buffered event count."""
        return {
            'enabled': self.enabled,
            'latest': asdict(self._latest),
            'buffered_events': len(self._events),
            'max_events': self.max_events,
        }

    def drain_events(self) -> List[Dict[str, Any]]:
        """Drain and return all buffered events.

        Merge point: a future HTTP/SSE/WebSocket adaptor can call this method
        and forward events to a Whisperflowactions-style HUD client.
        """
        drained = [asdict(event) for event in self._events]
        self._events.clear()
        return drained
