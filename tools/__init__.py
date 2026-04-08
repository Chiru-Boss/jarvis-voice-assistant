"""Tools package – builds and returns a fully-populated ToolRegistry."""

from __future__ import annotations

from core.tool_registry import ToolRegistry


def build_registry() -> ToolRegistry:
    """Instantiate a ToolRegistry and register all available tool modules.

    Returns
    -------
    ToolRegistry
        Registry pre-populated with laptop control, system, web API,
        knowledge base, and home automation tools.
    """
    from tools import (
        home_automation,
        knowledge_base,
        laptop_control,
        system_tools,
        web_apis,
    )

    registry = ToolRegistry()
    laptop_control.register_tools(registry)
    system_tools.register_tools(registry)
    web_apis.register_tools(registry)
    knowledge_base.register_tools(registry)
    home_automation.register_tools(registry)
    return registry
