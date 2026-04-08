"""Laptop Control Tools – open applications, run shell commands, file operations."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

from core.tool_registry import ToolRegistry

# Maximum characters returned when reading a file (prevents overflowing LLM context).
MAX_FILE_READ_CHARS = 2000


# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------

def open_application(app_name: str) -> str:
    """Open an application by name or path."""
    system = platform.system()
    try:
        if system == 'Windows':
            os.startfile(app_name)  # type: ignore[attr-defined]
        elif system == 'Darwin':
            subprocess.Popen(['open', '-a', app_name])
        else:  # Linux / other
            subprocess.Popen([app_name])
        return f"✅ Opened '{app_name}' successfully."
    except Exception as exc:
        return f"❌ Could not open '{app_name}': {exc}"


def execute_command(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output or '(No output)'
    except subprocess.TimeoutExpired:
        return f"❌ Command timed out after {timeout}s."
    except Exception as exc:
        return f"❌ Command failed: {exc}"


def file_operations(
    operation: str,
    path: str,
    content: str = '',
    destination: str = '',
) -> str:
    """Perform a file operation: create, read, move, or delete."""
    try:
        if operation == 'create':
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(content)
            return f"✅ Created '{path}'."

        elif operation == 'read':
            with open(path, 'r', encoding='utf-8') as fh:
                text = fh.read()
            # Limit output so it fits comfortably inside an LLM context.
            return text[:MAX_FILE_READ_CHARS] if len(text) > MAX_FILE_READ_CHARS else text

        elif operation == 'move':
            if not destination:
                return "❌ 'destination' is required for the move operation."
            shutil.move(path, destination)
            return f"✅ Moved '{path}' to '{destination}'."

        elif operation == 'delete':
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            else:
                return f"❌ Path not found: '{path}'."
            return f"✅ Deleted '{path}'."

        else:
            return f"❌ Unknown operation '{operation}'. Use: create, read, move, delete."

    except Exception as exc:
        return f"❌ File operation failed: {exc}"


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def register_tools(registry: ToolRegistry) -> None:
    """Register all laptop control tools with *registry*."""

    registry.register(
        name='open_application',
        description=(
            'Open an application by name (e.g. "Chrome", "VS Code", "Notepad"). '
            'On Windows uses os.startfile; on macOS uses "open -a"; on Linux launches directly.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'app_name': {
                    'type': 'string',
                    'description': 'Application name or full executable path.',
                },
            },
            'required': ['app_name'],
        },
        func=open_application,
        safe=True,
    )

    registry.register(
        name='execute_command',
        description=(
            'Execute a shell command and return its stdout/stderr output. '
            'Use cautiously – this has direct access to the operating system.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'Shell command to execute.',
                },
                'timeout': {
                    'type': 'integer',
                    'description': 'Maximum seconds to wait for the command (default 30).',
                },
            },
            'required': ['command'],
        },
        func=execute_command,
        safe=False,
        requires_approval=True,
    )

    registry.register(
        name='file_operations',
        description=(
            'Perform file system operations: '
            'create (write a new file), read (return file contents), '
            'move (rename/relocate), or delete a file or directory.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'operation': {
                    'type': 'string',
                    'enum': ['create', 'read', 'move', 'delete'],
                    'description': 'Operation to perform.',
                },
                'path': {
                    'type': 'string',
                    'description': 'Source file or directory path.',
                },
                'content': {
                    'type': 'string',
                    'description': 'Text content to write (required for create).',
                },
                'destination': {
                    'type': 'string',
                    'description': 'Destination path (required for move).',
                },
            },
            'required': ['operation', 'path'],
        },
        func=file_operations,
        safe=False,
        requires_approval=True,
    )
