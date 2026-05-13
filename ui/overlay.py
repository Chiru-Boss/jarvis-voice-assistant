from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class OverlayUI:
    """Minimal streaming status overlay (Tkinter-backed when available)."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self._window = None
        self._label = None
        self._state = 'idle'
        self._detail = 'Ready'

    def start(self) -> bool:
        if not self.enabled:
            return False
        try:
            import tkinter as tk
        except ImportError as exc:
            logger.debug('Tkinter unavailable for overlay: %s', exc)
            return False

        self._window = tk.Tk()
        self._window.title('JARVIS HUD')
        self._window.geometry('420x120+20+20')
        self._window.attributes('-topmost', True)
        self._window.configure(bg='black')
        self._label = tk.Label(
            self._window,
            text=self.render_text(),
            fg='#00ffaa',
            bg='black',
            justify='left',
            anchor='w',
            font=('Consolas', 11),
        )
        self._label.pack(fill='both', expand=True, padx=12, pady=12)
        self._window.update_idletasks()
        self._window.update()
        return True

    def stop(self) -> None:
        if self._window is not None:
            self._window.destroy()
        self._window = None
        self._label = None

    def update_state(self, state: str, detail: str = '', stream: Optional[str] = None) -> None:
        self._state = state
        if detail:
            self._detail = detail
        if stream:
            base_detail = self._detail if self._detail else ''
            self._detail = f'{base_detail}\n{stream}' if base_detail else stream
        if self._label is not None:
            self._label.configure(text=self.render_text())
            self._window.update_idletasks()
            self._window.update()

    def render_text(self) -> str:
        return f'JARVIS v3\nState: {self._state}\n{self._detail}'
