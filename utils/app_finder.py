"""App Finder – locate application executables on the host operating system.

Provides a small database of common application paths across Windows, macOS,
and Linux so that :class:`~core.app_controller.AppController` can launch apps
reliably without relying on the shell's PATH alone.
"""

from __future__ import annotations

import os
import platform
import shutil
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Known application paths per platform
# ---------------------------------------------------------------------------
# Each entry maps a normalised alias (lowercase, no spaces) to a dict of
# platform → list-of-candidate-paths (checked in order).
# On Windows, paths may use LOCALAPPDATA / APPDATA environment variables.

def _local_app_data() -> str:
    return os.environ.get('LOCALAPPDATA', '')


def _program_files() -> str:
    return os.environ.get('PROGRAMFILES', r'C:\Program Files')


def _program_files_x86() -> str:
    return os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)')


# Lazily evaluated so the env vars are read at call time, not import time.
def _build_app_db() -> Dict[str, Dict[str, List[str]]]:
    lad = _local_app_data()
    pf = _program_files()
    pf86 = _program_files_x86()

    return {
        'brave': {
            'windows': [
                os.path.join(lad, r'BraveSoftware\Brave-Browser\Application\brave.exe'),
                os.path.join(pf, r'BraveSoftware\Brave-Browser\Application\brave.exe'),
                os.path.join(pf86, r'BraveSoftware\Brave-Browser\Application\brave.exe'),
            ],
            'darwin': ['/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'],
            'linux': ['brave-browser', 'brave'],
        },
        'chrome': {
            'windows': [
                os.path.join(lad, r'Google\Chrome\Application\chrome.exe'),
                os.path.join(pf, r'Google\Chrome\Application\chrome.exe'),
                os.path.join(pf86, r'Google\Chrome\Application\chrome.exe'),
            ],
            'darwin': ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'],
            'linux': ['google-chrome', 'google-chrome-stable', 'chromium-browser', 'chromium'],
        },
        'firefox': {
            'windows': [
                os.path.join(pf, r'Mozilla Firefox\firefox.exe'),
                os.path.join(pf86, r'Mozilla Firefox\firefox.exe'),
            ],
            'darwin': ['/Applications/Firefox.app/Contents/MacOS/firefox'],
            'linux': ['firefox', 'firefox-esr'],
        },
        'edge': {
            'windows': [
                os.path.join(pf86, r'Microsoft\Edge\Application\msedge.exe'),
                os.path.join(pf, r'Microsoft\Edge\Application\msedge.exe'),
            ],
            'darwin': ['/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'],
            'linux': ['microsoft-edge', 'microsoft-edge-stable'],
        },
        'opera': {
            'windows': [
                os.path.join(lad, r'Programs\Opera\launcher.exe'),
                os.path.join(pf, r'Opera\launcher.exe'),
            ],
            'darwin': ['/Applications/Opera.app/Contents/MacOS/Opera'],
            'linux': ['opera'],
        },
        'vscode': {
            'windows': [
                os.path.join(lad, r'Programs\Microsoft VS Code\Code.exe'),
                os.path.join(pf, r'Microsoft VS Code\Code.exe'),
            ],
            'darwin': ['/Applications/Visual Studio Code.app/Contents/MacOS/Electron'],
            'linux': ['code', 'code-oss'],
        },
        'notepad': {
            'windows': [r'C:\Windows\System32\notepad.exe'],
            'darwin': [],
            'linux': ['gedit', 'kate', 'mousepad'],
        },
        'calculator': {
            'windows': [r'C:\Windows\System32\calc.exe'],
            'darwin': ['/Applications/Calculator.app/Contents/MacOS/Calculator'],
            'linux': ['gnome-calculator', 'kcalc', 'bc'],
        },
        'vlc': {
            'windows': [
                os.path.join(pf, r'VideoLAN\VLC\vlc.exe'),
                os.path.join(pf86, r'VideoLAN\VLC\vlc.exe'),
            ],
            'darwin': ['/Applications/VLC.app/Contents/MacOS/VLC'],
            'linux': ['vlc'],
        },
        'spotify': {
            'windows': [
                os.path.join(lad, r'Spotify\Spotify.exe'),
            ],
            'darwin': ['/Applications/Spotify.app/Contents/MacOS/Spotify'],
            'linux': ['spotify'],
        },
        'discord': {
            'windows': [
                os.path.join(lad, r'Discord\Update.exe'),
            ],
            'darwin': ['/Applications/Discord.app/Contents/MacOS/Discord'],
            'linux': ['discord'],
        },
        'slack': {
            'windows': [
                os.path.join(lad, r'slack\slack.exe'),
            ],
            'darwin': ['/Applications/Slack.app/Contents/MacOS/Slack'],
            'linux': ['slack'],
        },
        'terminal': {
            'windows': [r'C:\Windows\System32\cmd.exe'],
            'darwin': ['/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal'],
            'linux': ['gnome-terminal', 'konsole', 'xterm'],
        },
        'explorer': {
            'windows': [r'C:\Windows\explorer.exe'],
            'darwin': [],
            'linux': ['nautilus', 'dolphin', 'thunar'],
        },
    }


# Aliases to normalise user input to canonical keys above.
_ALIAS_MAP: Dict[str, str] = {
    'bravebrowser': 'brave',
    'googlechrome': 'chrome',
    'microsoftedge': 'edge',
    'msedge': 'edge',
    'visualstudiocode': 'vscode',
    'vs code': 'vscode',
    'vs': 'vscode',
    'notepad++': 'notepadplusplus',
    'cmd': 'terminal',
    'commandprompt': 'terminal',
}

# Names that identify web browsers (used for smart search routing).
BROWSER_NAMES: frozenset = frozenset(
    ['brave', 'chrome', 'firefox', 'edge', 'opera', 'safari']
)


def _normalise(name: str) -> str:
    """Normalise an app name to the canonical lookup key."""
    key = name.lower().strip().replace(' ', '').replace('-', '').replace('_', '')
    return _ALIAS_MAP.get(key, key)


def find_app_path(app_name: str) -> Optional[str]:
    """Return the full executable path for *app_name*, or ``None`` if not found.

    Checks the built-in database first, then falls back to :func:`shutil.which`.
    """
    system = platform.system().lower()  # 'windows', 'darwin', 'linux'
    key = _normalise(app_name)
    db = _build_app_db()

    candidates = db.get(key, {}).get(system, [])
    for candidate in candidates:
        # On Linux the candidate might be just a command name (no path sep).
        if os.sep in candidate or candidate.startswith('/'):
            if os.path.isfile(candidate):
                return candidate
        else:
            # Try shutil.which for plain command names.
            found = shutil.which(candidate)
            if found:
                return found

    # Fallback: try shutil.which with the raw name and common variants.
    for attempt in (app_name, app_name.lower(), key):
        found = shutil.which(attempt)
        if found:
            return found

    return None


def is_browser(app_name: str) -> bool:
    """Return ``True`` if *app_name* refers to a web browser."""
    key = _normalise(app_name)
    return key in BROWSER_NAMES
