from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.browser_executor import BrowserExecutor


class TestBrowserExecutor(unittest.TestCase):
    def test_disabled_browser_executor(self):
        executor = BrowserExecutor(enabled=False, use_playwright=True)
        result = executor.run_task(url='https://example.com', action='navigate')
        self.assertEqual(result['status'], 'disabled')

    def test_playwright_flag_off(self):
        executor = BrowserExecutor(enabled=True, use_playwright=False)
        result = executor.run_task(url='https://example.com', action='navigate')
        self.assertEqual(result['status'], 'disabled')

    def test_candidate_selectors_include_semantic_patterns(self):
        selectors = BrowserExecutor._candidate_selectors('Search')
        self.assertTrue(any('data-testid' in s for s in selectors))
        self.assertTrue(any('aria-label' in s for s in selectors))
        self.assertTrue(any('text=' in s for s in selectors))

    def test_strategy_cache_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / 'strategies.json'
            executor = BrowserExecutor(cache_path=str(cache_path))
            executor._store_strategy('Search', 'input[name="q"]')

            reloaded = BrowserExecutor(cache_path=str(cache_path))
            self.assertEqual(reloaded._resolve_selector('Search'), 'input[name="q"]')


if __name__ == '__main__':
    unittest.main()
