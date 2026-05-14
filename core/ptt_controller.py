"""Push-to-talk controller (Ctrl+Space by default)."""

from __future__ import annotations

from typing import Any, Callable, Optional, Set


class PushToTalkController:
    def __init__(
        self,
        *,
        enabled: bool = False,
        hotkey: str = 'ctrl+space',
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
    ):
        self.enabled = enabled
        self.hotkey = hotkey
        self._on_start = on_start
        self._on_stop = on_stop
        self._pressed: Set[str] = set()
        self._listening = False
        self._listener = None

    @property
    def is_listening(self) -> bool:
        return self._listening

    def key_pressed(self, key_name: str) -> None:
        self._pressed.add(key_name.lower())
        self._update_state()

    def key_released(self, key_name: str) -> None:
        self._pressed.discard(key_name.lower())
        self._update_state()

    def _update_state(self) -> None:
        should_listen = 'ctrl' in self._pressed and 'space' in self._pressed
        if should_listen and not self._listening:
            self._listening = True
            if self._on_start:
                self._on_start()
        elif not should_listen and self._listening:
            self._listening = False
            if self._on_stop:
                self._on_stop()

    def start(self) -> bool:
        """Start keyboard hotkey listener.

        Returns True when the listener starts successfully.
        Returns False if PTT is disabled or pynput is unavailable.
        """
        if not self.enabled:
            return False
        try:
            from pynput import keyboard  # type: ignore
        except Exception:
            return False

        def normalize(key: Any) -> str:
            text = str(key).lower()
            if 'space' in text:
                return 'space'
            if 'ctrl' in text:
                return 'ctrl'
            return text

        self._listener = keyboard.Listener(
            on_press=lambda key: self.key_pressed(normalize(key)),
            on_release=lambda key: self.key_released(normalize(key)),
        )
        self._listener.start()
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._pressed.clear()
        if self._listening:
            self._listening = False
            if self._on_stop:
                self._on_stop()
