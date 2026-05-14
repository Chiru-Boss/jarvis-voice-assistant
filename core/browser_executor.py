"""High-level browser execution wrapper for orchestration layer."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from executors.browser_executor import BrowserExecutor


class BrowserAutomationExecutor:
    def __init__(self, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = os.getenv('BROWSER_AUTOMATION_ENABLED', 'false').lower() == 'true'
        timeout = int(os.getenv('BROWSER_TIMEOUT', '30'))
        self._executor = BrowserExecutor(enabled=enabled, timeout_seconds=timeout)

    @property
    def enabled(self) -> bool:
        return self._executor.enabled

    def execute_task(self, instruction: str, *, page_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._executor.execute(instruction, page_snapshot=page_snapshot)
