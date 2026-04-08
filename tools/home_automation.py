"""Home Automation Tools – smart device control (placeholder for future integration).

These tools provide the API surface for home automation.  They return
informative placeholder responses until a real smart-home hub (e.g. Home
Assistant, SmartThings) is connected via the ``HOME_AUTOMATION_URL``
environment variable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

from core.tool_registry import ToolRegistry

_REQUEST_TIMEOUT = 10


def _hub_url() -> str:
    """Return the configured home automation hub base URL (empty = placeholder mode)."""
    return os.getenv('HOME_AUTOMATION_URL', '').rstrip('/')


def _hub_post(path: str, payload: Dict[str, Any]) -> str:
    """POST *payload* to the hub at *path* and return the response text."""
    url = f"{_hub_url()}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
        return resp.text
    except Exception as exc:
        return f"❌ Hub request failed: {exc}"


# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------

def control_lights(action: str, room: str = 'all', brightness: int = 100) -> str:
    """Control smart lights in a room."""
    hub = _hub_url()
    if hub:
        return _hub_post('/lights/control', {'action': action, 'room': room, 'brightness': brightness})
    return (
        f"[Home Automation – placeholder] Lights in '{room}': "
        f"action='{action}', brightness={brightness}%. "
        "Set HOME_AUTOMATION_URL to enable real smart-home control."
    )


def control_temperature(temperature_c: float, room: str = 'all') -> str:
    """Set the thermostat target temperature."""
    hub = _hub_url()
    if hub:
        return _hub_post('/thermostat/set', {'temperature_c': temperature_c, 'room': room})
    return (
        f"[Home Automation – placeholder] Thermostat in '{room}' → {temperature_c}°C. "
        "Set HOME_AUTOMATION_URL to enable real smart-home control."
    )


def control_devices(device: str, action: str) -> str:
    """Send an action command to a named smart device."""
    hub = _hub_url()
    if hub:
        return _hub_post('/devices/control', {'device': device, 'action': action})
    return (
        f"[Home Automation – placeholder] Device '{device}': action='{action}'. "
        "Set HOME_AUTOMATION_URL to enable real smart-home control."
    )


def get_home_status() -> Dict[str, Any]:
    """Return the current state of all smart home devices."""
    hub = _hub_url()
    if hub:
        try:
            resp = requests.get(f"{hub}/status", timeout=_REQUEST_TIMEOUT)
            return resp.json()
        except Exception as exc:
            return {'error': str(exc)}
    return {
        'status': 'placeholder',
        'message': 'Set HOME_AUTOMATION_URL to connect your smart-home hub.',
        'devices': [],
    }


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def register_tools(registry: ToolRegistry) -> None:
    """Register all home automation tools with *registry*."""

    registry.register(
        name='control_lights',
        description=(
            'Control smart lights: turn on, turn off, or dim them in a specific room. '
            'Requires HOME_AUTOMATION_URL to be set for real integration.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'action': {
                    'type': 'string',
                    'enum': ['on', 'off', 'dim'],
                    'description': 'Light action to perform.',
                },
                'room': {
                    'type': 'string',
                    'description': 'Room name or "all" for every room.',
                },
                'brightness': {
                    'type': 'integer',
                    'description': 'Brightness level 0–100 (used when action is "dim").',
                },
            },
            'required': ['action'],
        },
        func=control_lights,
        safe=True,
    )

    registry.register(
        name='control_temperature',
        description=(
            'Set the smart thermostat to a target temperature in a specific room. '
            'Requires HOME_AUTOMATION_URL to be set for real integration.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'temperature_c': {
                    'type': 'number',
                    'description': 'Target temperature in degrees Celsius.',
                },
                'room': {
                    'type': 'string',
                    'description': 'Room name or "all".',
                },
            },
            'required': ['temperature_c'],
        },
        func=control_temperature,
        safe=True,
    )

    registry.register(
        name='control_devices',
        description=(
            'Send an action command to a named smart home device. '
            'Requires HOME_AUTOMATION_URL to be set for real integration.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'device': {
                    'type': 'string',
                    'description': 'Smart device name, e.g. "TV", "coffee maker".',
                },
                'action': {
                    'type': 'string',
                    'description': 'Action to perform, e.g. "on", "off", "start".',
                },
            },
            'required': ['device', 'action'],
        },
        func=control_devices,
        safe=True,
    )

    registry.register(
        name='get_home_status',
        description=(
            'Get the current status of all smart home devices. '
            'Requires HOME_AUTOMATION_URL to be set for real integration.'
        ),
        parameters={
            'type': 'object',
            'properties': {},
        },
        func=get_home_status,
        safe=True,
    )
