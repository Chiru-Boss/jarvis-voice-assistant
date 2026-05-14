"""Playwright-backed browser task executor (optional)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from executors.dom_navigator import DOMNavigator


class BrowserExecutor:
    def __init__(self, *, enabled: bool = False, timeout_seconds: int = 30, navigator: Optional[DOMNavigator] = None):
        self.enabled = enabled
        self.timeout_seconds = int(timeout_seconds)
        self.navigator = navigator or DOMNavigator()

    def execute(self, instruction: str, *, page_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {'ok': False, 'message': 'Browser automation disabled'}

        # Offline-safe path for tests and dry-runs.
        if page_snapshot is not None:
            target = self.navigator.best_match(page_snapshot.get('elements', []), instruction)
            if not target:
                return {'ok': False, 'message': 'No matching interactive element found'}
            return {'ok': True, 'message': f"Selected element '{target.get('text', '')}'", 'target': target}

        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception as exc:
            return {'ok': False, 'message': f'Playwright unavailable: {exc}'}

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto('about:blank', wait_until='domcontentloaded', timeout=self.timeout_seconds * 1000)
            browser.close()
        return {'ok': True, 'message': 'Browser automation bootstrap completed'}
