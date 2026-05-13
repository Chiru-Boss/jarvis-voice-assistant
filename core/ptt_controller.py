from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class PushToTalkController:
    """Ctrl+Space push-to-talk hotkey controller."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        hotkey: str = '<ctrl>+<space>',
        on_toggle: Optional[Callable[[], None]] = None,
    ) -> None:
        self.enabled = enabled
        self.hotkey = hotkey
        self.on_toggle = on_toggle
        self._listener = None

    def start(self) -> bool:
        if not self.enabled:
            return False
        try:
            from pynput import keyboard  # type: ignore
        except ImportError as exc:
            logger.debug('pynput unavailable for push-to-talk: %s', exc)
            return False

        def _wrapped_toggle() -> None:
            if self.on_toggle:
                self.on_toggle()

        self._listener = keyboard.GlobalHotKeys({self.hotkey: _wrapped_toggle})
        self._listener.start()
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def trigger(self) -> None:
        if self.on_toggle:
            self.on_toggle()
