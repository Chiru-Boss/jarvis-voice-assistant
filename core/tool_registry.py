"""Tool Registry – registers, discovers, and validates MCP tools."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class ToolDefinition:
    """Represents a single registered tool with its metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
        safe: bool = True,
        requires_approval: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema object
        self.func = func
        self.safe = safe
        self.requires_approval = requires_approval

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function-calling tool schema."""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters,
            },
        }


class ToolRegistry:
    """Manages tool registration and discovery for the MCP server."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
        safe: bool = True,
        requires_approval: bool = False,
    ) -> None:
        """Register a callable as a named tool.

        Parameters
        ----------
        name : str
            Unique tool identifier (used in LLM function calls).
        description : str
            Human/LLM-readable description of what the tool does.
        parameters : dict
            JSON Schema object describing the tool's parameters.
        func : callable
            The Python function that implements the tool.
        safe : bool
            Whether the tool is considered non-destructive.
        requires_approval : bool
            Whether the tool requires user confirmation before running.
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
            safe=safe,
            requires_approval=requires_approval,
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Return the ToolDefinition for *name*, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_openai_schemas(self) -> List[Dict[str, Any]]:
        """Return all tools formatted for OpenAI function calling."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute the tool named *name* with keyword arguments.

        Raises
        ------
        ValueError
            If *name* is not registered.
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry.")
        return tool.func(**kwargs)
