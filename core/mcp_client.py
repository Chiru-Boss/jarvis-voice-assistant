"""MCP Client – thin wrapper for in-process tool execution via MCPServer."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.mcp_server import MCPServer

logger = logging.getLogger(__name__)


class MCPClient:
    """Calls tools through the MCPServer and returns human-readable results.

    The client is intentionally lightweight: all heavy lifting (tool lookup,
    argument validation, execution) lives in :class:`MCPServer`.
    """

    def __init__(self, server: MCPServer) -> None:
        self._server = server

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute *tool_name* with *arguments* and return a result string.

        Parameters
        ----------
        tool_name : str
            The registered tool identifier.
        arguments : dict
            Keyword arguments to pass to the tool function.

        Returns
        -------
        str
            Human-readable result suitable for feeding back to the LLM.
        """
        logger.info("MCPClient: calling '%s' with %s", tool_name, arguments)
        result = self._server.execute_tool(tool_name, arguments)
        return self._server.format_tool_result(tool_name, result)

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return all registered tools in OpenAI function-calling schema format."""
        return self._server.registry.get_openai_schemas()
