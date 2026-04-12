"""Browser Automation – control web browsers via Selenium or pyautogui.

Two-tier strategy
-----------------
1. **Selenium** (rich control) – preferred when *selenium* is installed and a
   matching ChromeDriver is available.  Works with Brave and Chrome.
2. **pyautogui keyboard shortcuts** – fallback that works with any focused
   Chromium browser window using ``Ctrl+L`` → type query → ``Enter``.

Usage example::

    from core.browser_automation import BrowserAutomation
    ba = BrowserAutomation()
    ba.open_browser('brave')
    ba.search('Python tutorials')
    ba.close()
"""

from __future__ import annotations

import logging
import platform
import subprocess
import time
from typing import Optional

from utils.app_finder import find_app_path
from utils.window_manager import focus_window

logger = logging.getLogger(__name__)

# Seconds to wait after launching the browser before trying to interact.
_BROWSER_LAUNCH_WAIT = 3.0
# Seconds to wait for a page to start loading after pressing Enter.
_PAGE_LOAD_WAIT = 1.5
# Seconds to wait after focusing a window before sending keyboard input.
_FOCUS_SETTLE = 0.5


class BrowserAutomation:
    """Control a web browser for searches and navigation.

    The preferred backend is Selenium WebDriver (rich control).  When Selenium
    is unavailable or fails to initialise, the class falls back to pyautogui
    keyboard shortcuts (``Ctrl+L`` to focus the address bar, type, ``Enter``).
    """

    def __init__(self) -> None:
        self._driver: Optional[object] = None
        self._browser_name: str = ''

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_browser(self, browser_name: str = 'brave') -> str:
        """Launch *browser_name* and prepare it for automation.

        Returns a human-readable status string.
        """
        exe_path = find_app_path(browser_name)
        if not exe_path:
            return (
                f"❌ Could not find the '{browser_name}' executable. "
                "Make sure it is installed."
            )

        # Try Selenium first.
        if self._try_selenium(exe_path, browser_name):
            self._browser_name = browser_name
            return f"✅ Opened {browser_name} (Selenium mode)."

        # Fallback: launch via subprocess.
        try:
            system = platform.system()
            if system == 'Windows':
                subprocess.Popen(
                    [exe_path],
                    creationflags=subprocess.DETACHED_PROCESS,  # type: ignore[attr-defined]
                )
            elif system == 'Darwin':
                subprocess.Popen(['open', '-a', exe_path])
            else:
                subprocess.Popen([exe_path], start_new_session=True)

            time.sleep(_BROWSER_LAUNCH_WAIT)
            self._browser_name = browser_name
            return f"✅ Opened {browser_name}."
        except Exception as exc:
            return f"❌ Could not open {browser_name}: {exc}"

    def search(self, query: str, browser_name: str = '') -> str:
        """Perform a web search for *query* in the active browser.

        The address bar shortcut ``Ctrl+L`` works in Brave, Chrome, Firefox,
        and Edge.  The method also handles the Selenium path when a live
        driver session is available.
        """
        target = browser_name or self._browser_name or 'brave'

        # Try Selenium driver first.
        if self._driver is not None:
            result = self._selenium_search(query)
            if result.startswith('✅'):
                return result
            # Driver may have died – fall through to pyautogui.

        return self._pyautogui_search(query, target)

    def get_current_url(self) -> Optional[str]:
        """Return the current URL if a Selenium session is active."""
        if self._driver is not None:
            try:
                return self._driver.current_url  # type: ignore[union-attr]
            except Exception:
                pass
        return None

    def close(self) -> str:
        """Close the browser (Selenium session only)."""
        if self._driver is not None:
            try:
                self._driver.quit()  # type: ignore[union-attr]
                self._driver = None
                return "✅ Browser closed."
            except Exception as exc:
                return f"❌ Could not close browser: {exc}"
        return "ℹ️ No active Selenium session."

    # ------------------------------------------------------------------
    # Selenium helpers
    # ------------------------------------------------------------------

    def _try_selenium(self, exe_path: str, browser_name: str) -> bool:
        """Attempt to start a Selenium WebDriver session. Returns True on success."""
        try:
            from selenium import webdriver  # type: ignore
            from selenium.webdriver.chrome.options import Options  # type: ignore

            options = Options()
            options.binary_location = exe_path
            options.add_argument('--no-first-run')
            options.add_argument('--no-default-browser-check')
            options.add_argument('--disable-features=TranslateUI')

            # Try webdriver-manager for automatic ChromeDriver download.
            try:
                from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
                from selenium.webdriver.chrome.service import Service  # type: ignore

                service = Service(ChromeDriverManager().install())
                self._driver = webdriver.Chrome(service=service, options=options)
                return True
            except Exception:
                pass

            # Try default ChromeDriver on PATH.
            self._driver = webdriver.Chrome(options=options)
            return True

        except Exception as exc:
            logger.debug('Selenium init failed (%s): %s', browser_name, exc)
            self._driver = None
            return False

    def _selenium_search(self, query: str) -> str:
        """Navigate to a Google search using the active Selenium driver."""
        try:
            search_url = (
                f"https://www.google.com/search?q={query.replace(' ', '+')}"
            )
            self._driver.get(search_url)  # type: ignore[union-attr]
            time.sleep(_PAGE_LOAD_WAIT)
            return f"✅ Searched for '{query}'."
        except Exception as exc:
            logger.debug('Selenium search failed: %s', exc)
            self._driver = None  # Driver may have crashed.
            return f"❌ Selenium search failed: {exc}"

    # ------------------------------------------------------------------
    # pyautogui fallback
    # ------------------------------------------------------------------

    def _pyautogui_search(self, query: str, browser_name: str) -> str:
        """Search by focusing the browser window and using Ctrl+L keyboard shortcut."""
        try:
            import pyautogui  # type: ignore
        except ImportError:
            return '❌ pyautogui not installed. Run: pip install pyautogui'

        import platform

        # macOS uses Cmd+L / Cmd+A; Windows and Linux use Ctrl+L / Ctrl+A.
        modifier = 'command' if platform.system() == 'Darwin' else 'ctrl'

        # Bring the browser window to the foreground.
        focused = focus_window(browser_name)
        if not focused:
            logger.debug(
                'Could not confirm %s window focus; attempting search anyway.',
                browser_name,
            )
        time.sleep(_FOCUS_SETTLE)

        try:
            # Ctrl+L / Cmd+L focuses the address / search bar in all major browsers.
            pyautogui.hotkey(modifier, 'l')
            time.sleep(0.3)
            # Select any existing content then type the new query.
            pyautogui.hotkey(modifier, 'a')
            time.sleep(0.1)
            pyautogui.write(query, interval=0.05)
            time.sleep(0.2)
            pyautogui.press('enter')
            time.sleep(_PAGE_LOAD_WAIT)
            return f"✅ Searched for '{query}' in {browser_name}."
        except Exception as exc:
            return f"❌ Browser search via keyboard failed: {exc}"
