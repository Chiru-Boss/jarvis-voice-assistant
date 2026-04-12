"""System Tools – CPU/RAM/disk info, screenshots, system control, and AI agent tools."""

from __future__ import annotations

import platform
import subprocess
from typing import Any, Dict, List, Optional

from core.tool_registry import ToolRegistry

# Lazy singleton references to the AI agent subsystems.
# These are set once from main.py via inject_agent() after the agent is created.
_adaptive_agent = None


def inject_agent(agent: Any) -> None:
    """Inject the shared AdaptiveAgent instance so tools can use it."""
    global _adaptive_agent
    _adaptive_agent = agent


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
# AI Agent tool implementations
# ------------------------------------------------------------------

def open_app(app_name: str) -> str:
    """Launch an application by name, reusing an existing window if already open."""
    if _adaptive_agent is None:
        # Fallback: use laptop_control's basic launcher.
        from tools.laptop_control import open_application
        return open_application(app_name)
    return _adaptive_agent.app_controller.open_app(app_name)


def close_app(app_name: str) -> str:
    """Close a running application by name."""
    if _adaptive_agent is None:
        return f"❌ Agent not initialised – cannot close '{app_name}'."
    return _adaptive_agent.app_controller.close_app(app_name)


def get_screen_content() -> Dict[str, Any]:
    """Capture the screen and return OCR text plus detected UI elements."""
    if _adaptive_agent is None:
        # Standalone fallback.
        from core.screen_vision import ScreenVision
        return ScreenVision().get_screen_content(force=True)
    return _adaptive_agent.vision.get_screen_content(force=True)


def click_element(description: str, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Click a UI element identified by description or absolute coordinates."""
    if _adaptive_agent is None:
        return '❌ Agent not initialised.'
    if x is not None and y is not None:
        return _adaptive_agent.app_controller.click_at(x, y)
    return _adaptive_agent.app_controller.click_element(description)


def type_text(text: str) -> str:
    """Type text into the currently focused window."""
    if _adaptive_agent is None:
        return '❌ Agent not initialised.'
    return _adaptive_agent.app_controller.type_text(text)


def get_app_list() -> List[str]:
    """Return a list of currently running application names."""
    if _adaptive_agent is None:
        from core.app_controller import AppController
        return AppController().get_running_apps()
    return _adaptive_agent.app_controller.get_running_apps()


def get_patterns() -> Dict[str, Any]:
    """Return learned user patterns (top apps, searches, workflows, predictions)."""
    if _adaptive_agent is None:
        return {'error': 'Agent not initialised. Patterns are not yet available.'}
    return _adaptive_agent.memory.get_all_patterns()


def predict_action(last_command: str = '') -> str:
    """Return predicted next actions based on learned patterns."""
    if _adaptive_agent is None:
        return '❌ Agent not initialised – no predictions available yet.'
    return _adaptive_agent.predictor.predict_action_text(
        last_command=last_command or None,
        max_suggestions=3,
    )


def press_key(key: str) -> str:
    """Press a keyboard key or hotkey combination (e.g. 'enter', 'ctrl+c')."""
    if _adaptive_agent is None:
        return '❌ Agent not initialised.'
    return _adaptive_agent.app_controller.press_key(key)


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

    # ── AI Agent tools ────────────────────────────────────────────────

    registry.register(
        name='open_app',
        description=(
            'Launch an application by name (e.g. "Brave", "VS Code", "Notepad"). '
            'If the app is already running its existing window is focused instead of '
            'opening a duplicate.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'app_name': {
                    'type': 'string',
                    'description': 'Application name or executable path.',
                },
            },
            'required': ['app_name'],
        },
        func=open_app,
        safe=True,
    )

    registry.register(
        name='close_app',
        description='Close a running application by its process name.',
        parameters={
            'type': 'object',
            'properties': {
                'app_name': {
                    'type': 'string',
                    'description': 'Application name to close (e.g. "chrome", "notepad").',
                },
            },
            'required': ['app_name'],
        },
        func=close_app,
        safe=False,
        requires_approval=True,
    )

    registry.register(
        name='get_screen_content',
        description=(
            'Capture the current screen and return OCR-extracted text plus a list of '
            'detected UI element positions. Use this to check what is currently visible '
            'on screen before deciding what to do next.'
        ),
        parameters={
            'type': 'object',
            'properties': {},
        },
        func=get_screen_content,
        safe=True,
    )

    registry.register(
        name='click_element',
        description=(
            'Click a UI element on screen. Provide a human-readable description and/or '
            'exact pixel coordinates (x, y).'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'description': {
                    'type': 'string',
                    'description': 'Human-readable description of the element to click.',
                },
                'x': {
                    'type': 'integer',
                    'description': 'Pixel x-coordinate (optional if description is enough).',
                },
                'y': {
                    'type': 'integer',
                    'description': 'Pixel y-coordinate (optional if description is enough).',
                },
            },
            'required': ['description'],
        },
        func=click_element,
        safe=True,
    )

    registry.register(
        name='type_text',
        description='Type a string of text into the currently focused window or input field.',
        parameters={
            'type': 'object',
            'properties': {
                'text': {
                    'type': 'string',
                    'description': 'Text to type.',
                },
            },
            'required': ['text'],
        },
        func=type_text,
        safe=True,
    )

    registry.register(
        name='press_key',
        description=(
            'Press a keyboard key or hotkey combination, '
            'e.g. "enter", "ctrl+c", "alt+tab", "backspace".'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'key': {
                    'type': 'string',
                    'description': 'Key name or hotkey combination (e.g. "ctrl+c").',
                },
            },
            'required': ['key'],
        },
        func=press_key,
        safe=True,
    )

    registry.register(
        name='get_app_list',
        description='Return a list of all currently running application/process names.',
        parameters={
            'type': 'object',
            'properties': {},
        },
        func=get_app_list,
        safe=True,
    )

    registry.register(
        name='get_patterns',
        description=(
            'Return JARVIS learned user patterns: most-used apps, frequent searches, '
            'common workflows, and time-of-day preferences.'
        ),
        parameters={
            'type': 'object',
            'properties': {},
        },
        func=get_patterns,
        safe=True,
    )

    registry.register(
        name='predict_action',
        description=(
            'Get AI predictions for the next most likely user action based on '
            'learned behavioural patterns. Optionally provide the last command for '
            'sequence-based prediction.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'last_command': {
                    'type': 'string',
                    'description': 'The most recent command for context (optional).',
                },
            },
        },
        func=predict_action,
        safe=True,
    )
