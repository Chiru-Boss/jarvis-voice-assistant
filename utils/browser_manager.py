"""Browser lifecycle management utility."""

from __future__ import annotations

from typing import Optional

from core.browser_executor import BrowserAutomationExecutor


class BrowserManager:
    def __init__(self, executor: Optional[BrowserAutomationExecutor] = None):
        self.executor = executor or BrowserAutomationExecutor()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> bool:
        self._active = self.executor.enabled
        return self._active

    def stop(self) -> None:
        self._active = False
