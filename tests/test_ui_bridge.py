"""Tests for core/ui_bridge.py."""

from __future__ import annotations

import unittest

from core.ui_bridge import UIBridge


class TestUIBridge(unittest.TestCase):
    def test_invalid_max_events_raises(self):
        with self.assertRaises(ValueError):
            UIBridge(enabled=True, max_events=0)

    def test_disabled_bridge_no_events(self):
        bridge = UIBridge(enabled=False, max_events=5)
        bridge.transition('thinking', 'planning')
        bridge.stream_update('token')
        self.assertEqual(bridge.snapshot()['buffered_events'], 0)
        self.assertEqual(bridge.drain_events(), [])

    def test_transitions_and_stream_updates_are_buffered(self):
        bridge = UIBridge(enabled=True, max_events=5)
        bridge.transition('listening', 'Waiting for wake word')
        bridge.stream_update('partial response')
        events = bridge.drain_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['state'], 'LISTENING')
        self.assertEqual(events[0]['stream'], False)
        self.assertEqual(events[1]['state'], 'STREAM')
        self.assertEqual(events[1]['stream'], True)

    def test_event_buffer_is_capped(self):
        bridge = UIBridge(enabled=True, max_events=2)
        bridge.transition('one', '')
        bridge.transition('two', '')
        bridge.transition('three', '')
        events = bridge.drain_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['state'], 'TWO')
        self.assertEqual(events[1]['state'], 'THREE')


if __name__ == '__main__':
    unittest.main()
