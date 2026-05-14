"""Tests for core/self_healer.py."""

from __future__ import annotations

import unittest

from core.self_healer import SelfHealer


class TestSelfHealer(unittest.TestCase):
    def test_diagnosis_is_plain_english(self):
        healer = SelfHealer(max_attempts=2)
        text = healer.diagnose('click send', 'button not found')
        self.assertIn('failed', text.lower())
        self.assertIn('button not found', text.lower())

    def test_recover_succeeds_after_retry(self):
        healer = SelfHealer(max_attempts=3)
        attempts = {'n': 0}

        def attempt():
            attempts['n'] += 1
            return attempts['n'] == 2

        result = healer.recover('click send', 'transient error', attempt)
        self.assertTrue(result.recovered)
        self.assertEqual(result.attempts, 2)


if __name__ == '__main__':
    unittest.main()
