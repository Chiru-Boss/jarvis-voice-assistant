from __future__ import annotations

import unittest

from core.self_healer import SelfHealer


class TestSelfHealer(unittest.TestCase):
    def setUp(self):
        self.healer = SelfHealer()

    def test_timeout_diagnosis(self):
        diagnosis = self.healer.diagnose_failure(error='Timeout waiting for selector', step='click')
        self.assertEqual(diagnosis['category'], 'timeout')

    def test_selector_diagnosis(self):
        diagnosis = self.healer.diagnose_failure(error='Element selector not found', step='click')
        self.assertEqual(diagnosis['category'], 'selector')

    def test_recovery_succeeds_on_second_attempt(self):
        state = {'n': 0}

        def flaky():
            state['n'] += 1
            if state['n'] < 2:
                raise RuntimeError('timeout')
            return 'ok'

        result = self.healer.recover(executor=flaky, max_attempts=3)
        self.assertTrue(result['ok'])
        self.assertEqual(result['attempt'], 2)

    def test_recovery_fails_after_max_attempts(self):
        def always_fail():
            raise RuntimeError('selector missing')

        result = self.healer.recover(executor=always_fail, max_attempts=2)
        self.assertFalse(result['ok'])
        self.assertEqual(result['attempt'], 2)
        self.assertEqual(len(result['attempts']), 2)


if __name__ == '__main__':
    unittest.main()
