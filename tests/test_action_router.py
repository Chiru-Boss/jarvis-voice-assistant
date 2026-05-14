"""Tests for core/action_router.py."""

from __future__ import annotations

import unittest

from core.action_router import ActionRouter


class TestActionRouter(unittest.TestCase):
    def setUp(self):
        self.router = ActionRouter()

    def test_routes_browser_when_capability_enabled(self):
        decision = self.router.decide('scroll the website and click checkout', capabilities={'browser': True})
        self.assertEqual(decision.route, 'browser')

    def test_routes_vision_when_verification_intent(self):
        decision = self.router.decide('verify if the message was sent', capabilities={'vision': True})
        self.assertEqual(decision.route, 'vision')

    def test_falls_back_to_adaptive_agent(self):
        decision = self.router.decide('hello jarvis', capabilities={})
        self.assertEqual(decision.route, 'adaptive_agent')


if __name__ == '__main__':
    unittest.main()
