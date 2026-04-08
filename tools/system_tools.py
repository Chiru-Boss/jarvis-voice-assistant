"""System Tools – CPU/RAM/disk info, screenshots, and system control."""

from __future__ import annotations

import platform
from typing import Any, Dict

from core.tool_registry import ToolRegistry


# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------

def get_system_info() -> Dict[str, Any]:
    """Return CPU, RAM, and disk usage statistics."""
    try:
        import psutil  # optional dependency
    except ImportError:
        return {'error': 'psutil is not installed. Run: pip install psutil'}

    try:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return {
            'cpu_percent': cpu,
            'ram_total_gb': round(mem.total / 1e9, 2),
            'ram_used_gb': round(mem.used / 1e9, 2),
            'ram_percent': mem.percent,
            'disk_total_gb': round(disk.total / 1e9, 2),
            'disk_used_gb': round(disk.used / 1e9, 2),
            'disk_percent': disk.percent,
            'platform': platform.platform(),
        }
    except Exception as exc:
        return {'error': str(exc)}


def take_screenshot(save_path: str = 'jarvis_screenshot.png') -> str:
    """Take a screenshot of the current screen and save it to *save_path*."""
    try:
        from PIL import ImageGrab  # optional dependency
    except ImportError:
        return '❌ Pillow is not installed. Run: pip install Pillow'

    try:
        img = ImageGrab.grab()
        img.save(save_path)
        return f"✅ Screenshot saved to '{save_path}'."
    except Exception as exc:
        return f"❌ Screenshot failed: {exc}"


def control_system(action: str) -> str:
    """Perform a system control action: sleep or lock the computer."""
    import subprocess  # noqa: PLC0415

    sys_name = platform.system()
    try:
        if action == 'sleep':
            if sys_name == 'Windows':
                subprocess.run(
                    ['rundll32.exe', 'powrprof.dll,SetSuspendState', '0', '1', '0'],
                    check=False,
                )
            elif sys_name == 'Darwin':
                subprocess.run(['pmset', 'sleepnow'], check=False)
            else:
                subprocess.run(['systemctl', 'suspend'], check=False)
            return '✅ System sleeping.'

        elif action == 'lock':
            if sys_name == 'Windows':
                import ctypes
                ctypes.windll.user32.LockWorkStation()  # type: ignore[attr-defined]
            elif sys_name == 'Darwin':
                subprocess.run(
                    [
                        'osascript',
                        '-e',
                        'tell application "System Events" to keystroke "q" '
                        'using {command down, control down, option down}',
                    ],
                    check=False,
                )
            else:
                subprocess.run(['loginctl', 'lock-session'], check=False)
            return '✅ System locked.'

        else:
            return f"❌ Unknown action '{action}'. Supported: sleep, lock."

    except Exception as exc:
        return f"❌ System control failed: {exc}"


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def register_tools(registry: ToolRegistry) -> None:
    """Register all system tools with *registry*."""

    registry.register(
        name='get_system_info',
        description=(
            'Get current system resource usage: CPU percentage, RAM used/total, '
            'disk used/total, and OS platform info.'
        ),
        parameters={
            'type': 'object',
            'properties': {},
        },
        func=get_system_info,
        safe=True,
    )

    registry.register(
        name='take_screenshot',
        description='Take a screenshot of the current screen and save it to a file.',
        parameters={
            'type': 'object',
            'properties': {
                'save_path': {
                    'type': 'string',
                    'description': 'File path to save the screenshot (default: jarvis_screenshot.png).',
                },
            },
        },
        func=take_screenshot,
        safe=True,
    )

    registry.register(
        name='control_system',
        description='Perform a system control action: put the computer to sleep or lock the screen.',
        parameters={
            'type': 'object',
            'properties': {
                'action': {
                    'type': 'string',
                    'enum': ['sleep', 'lock'],
                    'description': 'System action to perform.',
                },
            },
            'required': ['action'],
        },
        func=control_system,
        safe=False,
        requires_approval=True,
    )
