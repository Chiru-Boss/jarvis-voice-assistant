"""System Executor – safe execution of system commands and file operations.

Provides JARVIS with OS-level capabilities:

* Shell command execution with a blocked-command safety list.
* File read / write / delete / move / copy operations.
* Process management (list, kill).
* Directory navigation.
* Environment variable inspection.
* Action logging for auditing.
* Undo capability for recent destructive actions.

Safety first
------------
A hard-coded ``BLOCKED_COMMANDS`` set prevents execution of the most
dangerous operations (format, rm -rf /, registry edits, etc.).  Destructive
actions (delete, overwrite) are flagged and can optionally require external
confirmation.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum characters to return when reading a file.
_MAX_READ_CHARS = 4000

# Commands that JARVIS will never execute.
BLOCKED_COMMANDS: frozenset = frozenset(
    [
        'rm -rf /',
        'format c:',
        'format /',
        'mkfs',
        'dd if=/dev/zero',
        'del /f /s /q c:\\',
        ':(){ :|:& };:',  # fork bomb
        'shutdown /s /f',
        'reboot -f',
    ]
)

# Maximum number of undo steps to keep in memory.
_MAX_UNDO_STEPS = 10


class SystemExecutor:
    """Execute system-level operations with built-in safety checks.

    Attributes
    ----------
    _undo_stack : deque
        Recent reversible actions stored as ``(description, undo_fn)`` tuples.
    _action_log : list
        Chronological log of all executed actions for auditing.
    """

    def __init__(self) -> None:
        self._undo_stack: Deque[Tuple[str, Any]] = deque(maxlen=_MAX_UNDO_STEPS)
        self._action_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def execute_command(self, command: str, timeout: int = 30) -> str:
        """Execute *command* in a shell after safety checks.

        Returns stdout / stderr as a string.
        """
        # Safety check.
        cmd_lower = command.strip().lower()
        for blocked in BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return (
                    f"🚫 Blocked: the command matches a dangerous pattern "
                    f"('{blocked}'). JARVIS will not execute this."
                )

        self._log_action('execute_command', {'command': command})
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
            return f"❌ Command timed out after {timeout} s."
        except Exception as exc:
            return f"❌ Command failed: {exc}"

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def read_file(self, path: str) -> str:
        """Return the text content of *path* (truncated to avoid LLM overflow)."""
        self._log_action('read_file', {'path': path})
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                text = fh.read()
            if len(text) > _MAX_READ_CHARS:
                text = text[:_MAX_READ_CHARS] + '\n… (truncated)'
            return text
        except Exception as exc:
            return f"❌ Cannot read '{path}': {exc}"

    def write_file(self, path: str, content: str, overwrite: bool = True) -> str:
        """Write *content* to *path*.

        If the file already exists and *overwrite* is True, the original is
        backed up to enable undo.
        """
        backup: Optional[str] = None
        if os.path.exists(path) and overwrite:
            backup = path + '.jarvis_bak'
            shutil.copy2(path, backup)
            self._undo_stack.append(
                (f"Restore '{path}' from backup",
                 lambda p=path, b=backup: self._restore_backup(p, b))
            )

        self._log_action('write_file', {'path': path, 'chars': len(content)})
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(content)
            return f"✅ Written to '{path}'."
        except Exception as exc:
            return f"❌ Write failed: {exc}"

    def delete_file(self, path: str, require_confirm: bool = True) -> str:
        """Delete a file or directory.

        When *require_confirm* is True this method backs up the file before
        deleting so that :meth:`undo` can restore it.
        """
        if not os.path.exists(path):
            return f"⚠️ Path not found: '{path}'."

        backup = path + '.jarvis_deleted_bak'
        try:
            if os.path.isfile(path):
                shutil.copy2(path, backup)
                self._undo_stack.append(
                    (f"Restore deleted '{path}'",
                     lambda p=path, b=backup: self._restore_backup(p, b))
                )
                os.remove(path)
            else:
                shutil.copytree(path, backup)
                self._undo_stack.append(
                    (f"Restore deleted directory '{path}'",
                     lambda p=path, b=backup: shutil.copytree(b, p))
                )
                shutil.rmtree(path)
            self._log_action('delete_file', {'path': path})
            return f"✅ Deleted '{path}' (backup kept for undo)."
        except Exception as exc:
            return f"❌ Delete failed: {exc}"

    def copy_file(self, source: str, destination: str) -> str:
        """Copy *source* to *destination*."""
        self._log_action('copy_file', {'source': source, 'destination': destination})
        try:
            if os.path.isdir(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            return f"✅ Copied '{source}' → '{destination}'."
        except Exception as exc:
            return f"❌ Copy failed: {exc}"

    def move_file(self, source: str, destination: str) -> str:
        """Move / rename *source* to *destination*."""
        self._log_action('move_file', {'source': source, 'destination': destination})
        try:
            shutil.move(source, destination)
            self._undo_stack.append(
                (f"Move '{destination}' back to '{source}'",
                 lambda s=source, d=destination: shutil.move(d, s))
            )
            return f"✅ Moved '{source}' → '{destination}'."
        except Exception as exc:
            return f"❌ Move failed: {exc}"

    def list_directory(self, path: str = '.') -> str:
        """List the contents of *path*."""
        self._log_action('list_directory', {'path': path})
        try:
            entries = os.listdir(path)
            lines = []
            for entry in sorted(entries):
                full = os.path.join(path, entry)
                tag = '[DIR] ' if os.path.isdir(full) else '[FILE]'
                lines.append(f"  {tag} {entry}")
            return '\n'.join(lines) or '(empty directory)'
        except Exception as exc:
            return f"❌ Cannot list '{path}': {exc}"

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def list_processes(self) -> str:
        """Return a summary of the top 20 running processes by CPU usage."""
        try:
            import psutil  # type: ignore
        except ImportError:
            return '❌ psutil is not installed. Run: pip install psutil'

        procs = []
        for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                procs.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        procs.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
        lines = [f"  PID={p['pid']:6}  CPU={p.get('cpu_percent',0):5.1f}%"
                 f"  MEM={p.get('memory_percent',0):4.1f}%  {p['name']}"
                 for p in procs[:20]]
        return '\n'.join(lines)

    def kill_process(self, pid: int) -> str:
        """Terminate the process with the given *pid*."""
        try:
            import psutil  # type: ignore
            proc = psutil.Process(pid)
            proc.terminate()
            return f"✅ Terminated process {pid}."
        except ImportError:
            return '❌ psutil is not installed.'
        except Exception as exc:
            return f"❌ Could not kill process {pid}: {exc}"

    # ------------------------------------------------------------------
    # Environment / misc
    # ------------------------------------------------------------------

    def get_env_var(self, name: str) -> str:
        """Return the value of environment variable *name*."""
        value = os.environ.get(name)
        if value is None:
            return f"⚠️ Environment variable '{name}' is not set."
        return f"${name} = {value}"

    def get_current_directory(self) -> str:
        """Return the current working directory."""
        return os.getcwd()

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def undo(self) -> str:
        """Reverse the most recent reversible action."""
        if not self._undo_stack:
            return '⚠️ Nothing to undo.'
        description, undo_fn = self._undo_stack.pop()
        try:
            undo_fn()
            return f"✅ Undone: {description}."
        except Exception as exc:
            return f"❌ Undo failed: {exc}"

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def get_action_log(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Return the last *last_n* logged actions."""
        return self._action_log[-last_n:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_action(self, action: str, details: Dict[str, Any]) -> None:
        self._action_log.append({
            'timestamp': time.time(),
            'action': action,
            'details': details,
        })

    @staticmethod
    def _restore_backup(original: str, backup: str) -> None:
        """Restore *original* from *backup*, then remove the backup."""
        shutil.copy2(backup, original)
        os.remove(backup)
