"""Input Handler – real mouse clicks, keyboard typing, and key combinations.

Wraps *pyautogui* with JARVIS-friendly error handling and status messages.
All heavy imports are lazy so that the module loads even when pyautogui is
not installed.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _get_pyautogui():
    """Lazy import of pyautogui – raises ImportError if not installed."""
    try:
        import pyautogui  # type: ignore
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.05     # Small pause between actions
        return pyautogui
    except ImportError as exc:
        raise ImportError(
            "pyautogui is not installed. Run: pip install pyautogui"
        ) from exc


class InputHandler:
    """Real mouse and keyboard automation using pyautogui."""

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def click(self, x: int, y: int, button: str = 'left') -> str:
        """Click at absolute screen coordinates (*x*, *y*)."""
        try:
            pg = _get_pyautogui()
            pg.click(x, y, button=button)
            return f"✅ Clicked at ({x}, {y}) with {button} button."
        except ImportError as exc:
            return f"❌ {exc}"
        except Exception as exc:
            return f"❌ Click failed: {exc}"

    def double_click(self, x: int, y: int) -> str:
        """Double-click at screen coordinates (*x*, *y*)."""
        try:
            pg = _get_pyautogui()
            pg.doubleClick(x, y)
            return f"✅ Double-clicked at ({x}, {y})."
        except ImportError as exc:
            return f"❌ {exc}"
        except Exception as exc:
            return f"❌ Double-click failed: {exc}"

    def move_mouse(self, x: int, y: int, duration: float = 0.3) -> str:
        """Move the mouse smoothly to (*x*, *y*) over *duration* seconds."""
        try:
            pg = _get_pyautogui()
            pg.moveTo(x, y, duration=duration)
            return f"✅ Mouse moved to ({x}, {y})."
        except ImportError as exc:
            return f"❌ {exc}"
        except Exception as exc:
            return f"❌ Mouse move failed: {exc}"

    def scroll(self, x: int, y: int, clicks: int = 3) -> str:
        """Scroll at position (*x*, *y*) by *clicks* (positive = up)."""
        try:
            pg = _get_pyautogui()
            pg.scroll(clicks, x=x, y=y)
            return f"✅ Scrolled {clicks} clicks at ({x}, {y})."
        except ImportError as exc:
            return f"❌ {exc}"
        except Exception as exc:
            return f"❌ Scroll failed: {exc}"

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def type_text(self, text: str, interval: float = 0.05) -> str:
        """Type *text* into the currently focused window character by character."""
        try:
            pg = _get_pyautogui()
            pg.write(text, interval=interval)
            return f"✅ Typed: '{text}'."
        except ImportError as exc:
            return f"❌ {exc}"
        except Exception as exc:
            return f"❌ Typing failed: {exc}"

    def press_key(self, key: str) -> str:
        """Press a single key (e.g. ``'enter'``, ``'tab'``, ``'escape'``)."""
        try:
            pg = _get_pyautogui()
            pg.press(key)
            return f"✅ Pressed key: '{key}'."
        except ImportError as exc:
            return f"❌ {exc}"
        except Exception as exc:
            return f"❌ Key press failed: {exc}"

    def press_combination(self, *keys: str) -> str:
        """Press a key combination (e.g. ``'ctrl', 'l'`` for Ctrl+L)."""
        try:
            pg = _get_pyautogui()
            pg.hotkey(*keys)
            return f"✅ Pressed: {'+'.join(keys)}."
        except ImportError as exc:
            return f"❌ {exc}"
        except Exception as exc:
            return f"❌ Key combination failed: {exc}"

    def clear_and_type(self, text: str) -> str:
        """Select all existing text then type *text* (useful for address bars)."""
        try:
            import platform
            pg = _get_pyautogui()
            # macOS uses Cmd+A; Windows/Linux use Ctrl+A.
            modifier = 'command' if platform.system() == 'Darwin' else 'ctrl'
            pg.hotkey(modifier, 'a')
            time.sleep(0.1)
            pg.write(text, interval=0.05)
            return f"✅ Cleared field and typed: '{text}'."
        except ImportError as exc:
            return f"❌ {exc}"
        except Exception as exc:
            return f"❌ Clear-and-type failed: {exc}"
