"""Window Manager – bring application windows to the foreground.

Provides a cross-platform ``focus_window`` helper that JARVIS uses before
interacting with an application via keyboard/mouse automation.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Map friendly names → process names used when searching for windows.
_PROCESS_NAME_MAP: dict = {
    'brave': 'brave',
    'bravebrowser': 'brave',
    'chrome': 'chrome',
    'googlechrome': 'chrome',
    'firefox': 'firefox',
    'edge': 'msedge',
    'microsoftedge': 'msedge',
    'msedge': 'msedge',
    'opera': 'opera',
    'vscode': 'code',
    'visualstudiocode': 'code',
    'notepad': 'notepad',
    'calculator': 'calc',
    'explorer': 'explorer',
    'spotify': 'spotify',
    'discord': 'discord',
    'slack': 'slack',
}


def _normalise_name(app_name: str) -> str:
    key = app_name.lower().strip().replace(' ', '').replace('-', '')
    return _PROCESS_NAME_MAP.get(key, app_name.lower().replace(' ', ''))


def focus_window(app_name: str) -> bool:
    """Bring the window for *app_name* to the foreground.

    Returns ``True`` when focus was likely achieved (best-effort; no guarantee
    on all OS/WM combinations).
    """
    system = platform.system()
    try:
        if system == 'Windows':
            return _focus_windows(app_name)
        elif system == 'Darwin':
            return _focus_macos(app_name)
        else:
            return _focus_linux(app_name)
    except Exception as exc:
        logger.debug('focus_window(%s) failed: %s', app_name, exc)
        return False


# ---------------------------------------------------------------------------
# Platform implementations
# ---------------------------------------------------------------------------

_WIN32_POWERSHELL_TEMPLATE = r"""
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class JarvisWin32 {{
    [DllImport("user32.dll", SetLastError=true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}}
"@
$proc = Get-Process -Name "{proc_name}" -ErrorAction SilentlyContinue |
        Where-Object {{ $_.MainWindowHandle -ne 0 }} |
        Select-Object -First 1
if ($proc) {{
    [JarvisWin32]::ShowWindow($proc.MainWindowHandle, 9)  # SW_RESTORE
    [JarvisWin32]::SetForegroundWindow($proc.MainWindowHandle)
    Write-Output "focused"
}} else {{
    Write-Output "not_found"
}}
"""


def _focus_windows(app_name: str) -> bool:
    proc_name = _normalise_name(app_name)
    script = _WIN32_POWERSHELL_TEMPLATE.format(proc_name=proc_name)
    try:
        result = subprocess.run(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
            capture_output=True,
            text=True,
            timeout=6,
        )
        return 'focused' in result.stdout
    except Exception as exc:
        logger.debug('_focus_windows(%s) error: %s', app_name, exc)
        return False


def _focus_macos(app_name: str) -> bool:
    # Resolve friendly names to macOS application names.
    _macos_names = {
        'brave': 'Brave Browser',
        'chrome': 'Google Chrome',
        'firefox': 'Firefox',
        'edge': 'Microsoft Edge',
        'vscode': 'Visual Studio Code',
        'spotify': 'Spotify',
    }
    mac_name = _macos_names.get(_normalise_name(app_name), app_name)
    try:
        subprocess.run(
            ['osascript', '-e', f'tell application "{mac_name}" to activate'],
            timeout=4,
            check=False,
            capture_output=True,
        )
        return True
    except Exception as exc:
        logger.debug('_focus_macos(%s) error: %s', app_name, exc)
        return False


def _focus_linux(app_name: str) -> bool:
    # wmctrl can activate windows by name substring.
    proc_name = _normalise_name(app_name)
    try:
        result = subprocess.run(
            ['wmctrl', '-a', proc_name],
            timeout=3,
            capture_output=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        logger.debug('wmctrl not available on this Linux system.')
    except Exception as exc:
        logger.debug('_focus_linux(%s) error: %s', app_name, exc)
    return False
