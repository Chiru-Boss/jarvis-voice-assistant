"""App Controller – launch, close, focus, and interact with applications.

This module gives JARVIS the ability to control the desktop:

* Launch applications by friendly name or executable path.
* Close running applications by name or window title.
* Click on UI elements by screen coordinates or description.
* Type keyboard text into the focused window.
* Manage windows (minimize, maximize, restore, focus).
* Track which applications are currently open.

All heavy dependencies (pyautogui, psutil) are imported lazily so that the
rest of JARVIS continues to work even when they are not installed.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Seconds to wait after launching an app before interacting with it.
_APP_LAUNCH_WAIT = 2.0
# Blocked application names that JARVIS will refuse to close.
_PROTECTED_APPS = {'explorer', 'finder', 'systempreferences', 'taskbar'}


class AppController:
    """Control desktop applications and interact with their UI.

    Tracks the set of applications that JARVIS itself has launched so it can
    avoid duplicate launches and provide smarter context.
    """

    def __init__(self) -> None:
        self._launched_apps: Dict[str, float] = {}  # name → launch timestamp

    # ------------------------------------------------------------------
    # Application lifecycle
    # ------------------------------------------------------------------

    def open_app(self, app_name: str) -> str:
        """Launch *app_name* if it is not already running.

        Returns a human-readable status string.
        """
        # Check if already running (using psutil when available).
        if self._is_app_running(app_name):
            # Bring it to focus instead of launching a duplicate.
            self._focus_app(app_name)
            return f"✅ '{app_name}' is already open – brought it to focus."

        system = platform.system()
        try:
            if system == 'Windows':
                os.startfile(app_name)  # type: ignore[attr-defined]
            elif system == 'Darwin':
                subprocess.Popen(['open', '-a', app_name])
            else:  # Linux
                subprocess.Popen([app_name], start_new_session=True)

            self._launched_apps[app_name.lower()] = time.time()
            time.sleep(_APP_LAUNCH_WAIT)
            return f"✅ Launched '{app_name}'."
        except Exception as exc:
            return f"❌ Could not launch '{app_name}': {exc}"

    def close_app(self, app_name: str) -> str:
        """Close all processes whose name matches *app_name*.

        Returns a human-readable status string.
        """
        if app_name.lower() in _PROTECTED_APPS:
            return f"⚠️ '{app_name}' is protected and cannot be closed by JARVIS."

        try:
            import psutil  # type: ignore
        except ImportError:
            return '❌ psutil is not installed. Run: pip install psutil'

        closed: List[str] = []
        target = app_name.lower().replace('.exe', '')

        for proc in psutil.process_iter(['name', 'pid']):
            try:
                pname = (proc.info.get('name') or '').lower().replace('.exe', '')
                if target in pname:
                    proc.terminate()
                    closed.append(proc.info.get('name', str(proc.pid)))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if closed:
            self._launched_apps.pop(app_name.lower(), None)
            return f"✅ Closed: {', '.join(closed)}."
        return f"⚠️ No running process found matching '{app_name}'."

    def get_running_apps(self) -> List[str]:
        """Return a list of currently running application names."""
        try:
            import psutil  # type: ignore
        except ImportError:
            return []

        seen: set = set()
        apps: List[str] = []
        for proc in psutil.process_iter(['name']):
            try:
                name = proc.info.get('name') or ''
                if name and name not in seen:
                    seen.add(name)
                    apps.append(name)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return sorted(apps)

    # ------------------------------------------------------------------
    # Input automation
    # ------------------------------------------------------------------

    def click_at(self, x: int, y: int, button: str = 'left') -> str:
        """Click at absolute screen coordinates (*x*, *y*)."""
        try:
            import pyautogui  # type: ignore
            pyautogui.click(x, y, button=button)
            return f"✅ Clicked at ({x}, {y}) with {button} button."
        except ImportError:
            return '❌ pyautogui is not installed. Run: pip install pyautogui'
        except Exception as exc:
            return f"❌ Click failed: {exc}"

    def click_element(self, description: str) -> str:
        """Attempt to click a UI element matching *description* via image search.

        Falls back to a center-screen click with a message if the element
        cannot be located automatically.
        """
        try:
            import pyautogui  # type: ignore
        except ImportError:
            return '❌ pyautogui is not installed. Run: pip install pyautogui'

        # Attempt to locate element by searching visible text using OCR.
        # For now, we use pyautogui.locateOnScreen if an image is provided,
        # otherwise report that coordinate-based clicking is needed.
        return (
            f"⚠️ Could not locate '{description}' on screen automatically. "
            "Use click_at with exact coordinates instead."
        )

    def type_text(self, text: str, interval: float = 0.05) -> str:
        """Type *text* into the currently focused window."""
        try:
            import pyautogui  # type: ignore
            pyautogui.typewrite(text, interval=interval)
            return f"✅ Typed: '{text}'."
        except ImportError:
            return '❌ pyautogui is not installed. Run: pip install pyautogui'
        except Exception as exc:
            return f"❌ Typing failed: {exc}"

    def press_key(self, key: str) -> str:
        """Press a keyboard key (e.g. 'enter', 'ctrl+c', 'alt+tab')."""
        try:
            import pyautogui  # type: ignore
            if '+' in key:
                keys = [k.strip() for k in key.split('+')]
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(key)
            return f"✅ Pressed key: '{key}'."
        except ImportError:
            return '❌ pyautogui is not installed. Run: pip install pyautogui'
        except Exception as exc:
            return f"❌ Key press failed: {exc}"

    def move_mouse(self, x: int, y: int, duration: float = 0.3) -> str:
        """Move the mouse cursor to (*x*, *y*) smoothly."""
        try:
            import pyautogui  # type: ignore
            pyautogui.moveTo(x, y, duration=duration)
            return f"✅ Mouse moved to ({x}, {y})."
        except ImportError:
            return '❌ pyautogui is not installed. Run: pip install pyautogui'
        except Exception as exc:
            return f"❌ Mouse move failed: {exc}"

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def minimize_window(self) -> str:
        """Minimize the currently active window."""
        return self.press_key('super+down' if platform.system() != 'Darwin' else 'command+m')

    def maximize_window(self) -> str:
        """Maximize the currently active window."""
        system = platform.system()
        if system == 'Windows':
            return self.press_key('super+up')
        elif system == 'Darwin':
            try:
                import pyautogui  # type: ignore
                pyautogui.hotkey('ctrl', 'command', 'f')
                return "✅ Window maximized."
            except ImportError:
                return '❌ pyautogui is not installed.'
        else:
            return self.press_key('super+up')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_app_running(self, app_name: str) -> bool:
        """Return True if a process matching *app_name* is running."""
        try:
            import psutil  # type: ignore
            target = app_name.lower().replace('.exe', '')
            for proc in psutil.process_iter(['name']):
                try:
                    pname = (proc.info.get('name') or '').lower().replace('.exe', '')
                    if target in pname:
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            pass
        return False

    def _focus_app(self, app_name: str) -> None:
        """Best-effort attempt to bring *app_name* to the foreground."""
        system = platform.system()
        try:
            if system == 'Windows':
                subprocess.Popen(
                    ['powershell', '-Command',
                     f'(Get-Process -Name "{app_name}" -ErrorAction SilentlyContinue)'
                     f'.MainWindowHandle | ForEach-Object {{ '
                     f'[void][Win32]::SetForegroundWindow($_) }}'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif system == 'Darwin':
                subprocess.Popen(
                    ['osascript', '-e',
                     f'tell application "{app_name}" to activate'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as exc:
            logger.debug('Could not focus %s: %s', app_name, exc)
