"""Overlay HUD abstraction for real-time status updates."""

from __future__ import annotations

from typing import Dict

from ui.hud_state_manager import HUDStateManager


class OverlayHUD:
    def __init__(self, *, enabled: bool = False, position: str = 'top-right', opacity: float = 0.9):
        self.enabled = enabled
        self.position = position
        self.opacity = opacity
        self.state_manager = HUDStateManager()

    def update(self, state: str, detail: str = '') -> None:
        if not self.enabled:
            return
        self.state_manager.transition(state, detail)

    def status(self) -> Dict[str, object]:
        payload = self.state_manager.snapshot()
        payload.update({'enabled': self.enabled, 'position': self.position, 'opacity': self.opacity})
        return payload
