"""MCP Server – backend tool execution hub with optional HTTP/SSE endpoint."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

from core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPServer:
    """Manages tool execution and exposes an optional HTTP/SSE endpoint.

    In the default in-process mode the server is used entirely through
    :meth:`execute_tool`.  Call :meth:`start_http_server` to additionally
    expose an HTTP interface for external clients.
    """

    def __init__(self, registry: ToolRegistry, approval_mode: bool = False) -> None:
        self.registry = registry
        self.approval_mode = approval_mode
        self._http_server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a registered tool and return a standardised response dict.

        Returns
        -------
        dict
            ``{'success': True, 'result': ...}`` on success, or
            ``{'success': False, 'error': '...'}`` on failure.
        """
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            return {'success': False, 'error': f"Tool '{tool_name}' not found."}

        if self.approval_mode and tool.requires_approval:
            return {
                'success': False,
                'error': (
                    f"Tool '{tool_name}' requires explicit user approval "
                    f"(APPROVAL_MODE is enabled)."
                ),
                'requires_approval': True,
            }

        try:
            result = tool.func(**arguments)
            return {'success': True, 'result': result}
        except TypeError as exc:
            return {'success': False, 'error': f"Invalid arguments for '{tool_name}': {exc}"}
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", tool_name)
            return {'success': False, 'error': str(exc)}

    def format_tool_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Serialise a tool result to a string suitable for the LLM."""
        if result.get('success'):
            value = result.get('result', '')
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False, indent=2)
            return str(value)
        return f"Error from tool '{tool_name}': {result.get('error', 'Unknown error')}"

    # ------------------------------------------------------------------
    # Optional HTTP / SSE server
    # ------------------------------------------------------------------

    def start_http_server(self, host: str = '127.0.0.1', port: int = 8765) -> None:
        """Start an HTTP server with SSE support in a background daemon thread.

        Endpoints
        ---------
        GET  /tools      – list all registered tools (JSON)
        POST /execute    – execute a tool; body: ``{"tool": "...", "arguments": {...}}``
        POST /sse        – same as /execute but responds with ``text/event-stream``
        """
        registry = self.registry
        mcp_server = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path == '/tools':
                    self._send_json(registry.get_openai_schemas())
                else:
                    self._send_json({'error': 'Not found'}, code=404)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length) or b'{}')
                tool_name = body.get('tool', '')
                arguments = body.get('arguments', {})

                if self.path == '/execute':
                    result = mcp_server.execute_tool(tool_name, arguments)
                    self._send_json(result)

                elif self.path == '/sse':
                    result = mcp_server.execute_tool(tool_name, arguments)
                    event_data = json.dumps(result)
                    payload = f'data: {event_data}\n\n'.encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                else:
                    self._send_json({'error': 'Not found'}, code=404)

            def _send_json(self, obj: Any, code: int = 200) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: Any) -> None:  # silence noisy logs
                logger.debug(fmt, *args)

        self._http_server = HTTPServer((host, port), _Handler)
        self._server_thread = threading.Thread(
            target=self._http_server.serve_forever, daemon=True
        )
        self._server_thread.start()
        logger.info("MCP HTTP server started on http://%s:%d", host, port)
        print(f'🌐 MCP HTTP server running on http://{host}:{port}')

    def stop_http_server(self) -> None:
        """Shut down the background HTTP server (if running)."""
        if self._http_server:
            self._http_server.shutdown()
            self._http_server = None
            self._server_thread = None
