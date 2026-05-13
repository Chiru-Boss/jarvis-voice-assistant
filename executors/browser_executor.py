from __future__ import annotations

from typing import Dict

from core.browser_executor import BrowserExecutor, DEFAULT_BROWSER_URL


class BrowserTaskExecutor:
    """Specialized browser executor wrapper for orchestrator routing."""

    def __init__(self, browser: BrowserExecutor | None = None) -> None:
        self.browser = browser or BrowserExecutor()

    def execute_search(self, query: str) -> Dict[str, str]:
        return self.browser.run_task(
            url=DEFAULT_BROWSER_URL,
            action='type',
            target='Search',
            value=query,
        )
