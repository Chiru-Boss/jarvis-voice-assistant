"""System tray state holder (dependency-light placeholder)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrayStatus:
    state: str = 'idle'
    detail: str = 'JARVIS ready'


class TrayIndicator:
    def __init__(self, *, enabled: bool = False):
        self.enabled = enabled
        self._status = TrayStatus()

    def set_status(self, state: str, detail: str = '') -> None:
        if not self.enabled:
            return
        self._status = TrayStatus(state=state, detail=detail)

    def snapshot(self) -> TrayStatus:
        return self._status
