"""Tests for browser automation executors."""

from __future__ import annotations

import unittest

from core.browser_executor import BrowserAutomationExecutor
from executors.dom_navigator import DOMNavigator


class TestDOMNavigator(unittest.TestCase):
    def test_best_match_prefers_visible_interactive_with_matching_text(self):
        nav = DOMNavigator()
        elements = [
            {'tag': 'div', 'text': 'submit', 'visible': True, 'importance': 0.1},
            {'tag': 'button', 'text': 'Submit order', 'visible': True, 'importance': 0.5},
        ]
        best = nav.best_match(elements, 'submit')
        self.assertEqual(best['tag'], 'button')


class TestBrowserAutomationExecutor(unittest.TestCase):
    def test_disabled_executor_is_safe(self):
        executor = BrowserAutomationExecutor(enabled=False)
        result = executor.execute_task('click submit')
        self.assertFalse(result['ok'])

    def test_page_snapshot_selects_matching_element(self):
        executor = BrowserAutomationExecutor(enabled=True)
        result = executor.execute_task(
            'checkout',
            page_snapshot={
                'elements': [
                    {'tag': 'button', 'text': 'Checkout', 'visible': True, 'importance': 1.0},
                ]
            },
        )
        self.assertTrue(result['ok'])
        self.assertEqual(result['target']['text'], 'Checkout')


if __name__ == '__main__':
    unittest.main()
