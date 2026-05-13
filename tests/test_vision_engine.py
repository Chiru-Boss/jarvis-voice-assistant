from __future__ import annotations

import unittest

from core.vision_engine import VisionEngine


class TestVisionEngine(unittest.TestCase):
    def test_disabled_engine_returns_error(self):
        engine = VisionEngine(enabled=False)
        result = engine.analyze_screenshot(goal='open browser', image_bytes=b'fake')
        self.assertFalse(result['ok'])
        self.assertIn('disabled', result['error'].lower())

    def test_missing_api_key_returns_error(self):
        engine = VisionEngine(enabled=True, api_key='')
        result = engine.analyze_screenshot(goal='x', image_bytes=b'fake')
        self.assertFalse(result['ok'])
        self.assertIn('api_key', result['error'].lower())

    def test_goal_verification_parsing(self):
        engine = VisionEngine(enabled=True, api_key='test')
        engine._call_vision_model = lambda **_: 'UI looks correct. GOAL_VERIFIED: yes'  # type: ignore[method-assign]
        result = engine.verify_goal('search results visible', image_bytes=b'fake')
        self.assertTrue(result['ok'])
        self.assertTrue(result['goal_verified'])


if __name__ == '__main__':
    unittest.main()
