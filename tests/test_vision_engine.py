"""Tests for core/vision_engine.py."""

from __future__ import annotations

import unittest

from core.vision_engine import VisionConfig, VisionEngine


class TestVisionEngine(unittest.TestCase):
    def test_disabled_engine_returns_disabled_response(self):
        engine = VisionEngine(config=VisionConfig(enabled=False))
        result = engine.verify_goal('Was the message sent?', image_bytes=b'abc')
        self.assertFalse(result['enabled'])

    def test_missing_api_key_is_reported(self):
        engine = VisionEngine(config=VisionConfig(enabled=True, api_key=''))
        result = engine.verify_goal('Was the message sent?', image_bytes=b'abc')
        self.assertIn('missing', result['analysis'].lower())

    def test_structured_response_is_parsed(self):
        def fake_request(_payload):
            return {
                'content': '{"completed": true, "diagnosis": "Message is visible in sent chat.", "recovery_plan": ["No action needed"]}'
            }

        engine = VisionEngine(config=VisionConfig(enabled=True, api_key='x'), requester=fake_request)
        result = engine.verify_goal('Was the message sent?', image_bytes=b'img')
        self.assertTrue(result['completed'])
        self.assertEqual(result['recovery_plan'], ['No action needed'])


if __name__ == '__main__':
    unittest.main()
