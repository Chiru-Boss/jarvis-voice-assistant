"""Tests for core/chat_session.py – session deduplication manager."""

from __future__ import annotations

import time
import unittest

from core.chat_session import (
    ChatSessionManager,
    Session,
    is_confirmation_phrase,
)


# ---------------------------------------------------------------------------
# is_confirmation_phrase
# ---------------------------------------------------------------------------

class TestIsConfirmationPhrase(unittest.TestCase):
    """Verify which messages are recognised as bare confirmations."""

    # True positives – should be detected as confirmations
    _CONFIRMATIONS = [
        'yes',
        'Yeah',
        'yep',
        'ok',
        'okay',
        'sure',
        'confirmed',
        'confirm',
        'Confirming',
        'go ahead',
        'proceed',
        'start it',
        'do it',
        'merge done',
        'done merge',
        '@Copilot Accepted Confirmation: Confirm agent session',
        'Accepted Confirmation: Confirm agent session',
        'confirm agent session',
        'confirm session',
        'agent session',
        '  YES  ',
    ]

    # True negatives – substantive requests that must NOT be suppressed
    _REQUESTS = [
        'open Brave and search for Python tutorials',
        'please add a dark mode toggle to the settings panel',
        'run the full test suite and report failures',
        'what is the current CPU usage?',
        'show me the pattern database',
        'yes please open Visual Studio Code and run the tests',  # >12 words
    ]

    def test_detects_all_confirmation_phrases(self):
        for phrase in self._CONFIRMATIONS:
            with self.subTest(phrase=phrase):
                self.assertTrue(
                    is_confirmation_phrase(phrase),
                    msg=f"Expected '{phrase}' to be detected as a confirmation",
                )

    def test_does_not_flag_substantive_requests(self):
        for phrase in self._REQUESTS:
            with self.subTest(phrase=phrase):
                self.assertFalse(
                    is_confirmation_phrase(phrase),
                    msg=f"Expected '{phrase}' NOT to be detected as a confirmation",
                )


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

class TestSession(unittest.TestCase):
    def test_initial_status_is_in_progress(self):
        s = Session(session_id='abc', description='test', status='in_progress')
        self.assertEqual(s.status, 'in_progress')
        self.assertIsNone(s.completed_at)

    def test_complete_sets_status_and_timestamp(self):
        s = Session(session_id='abc', description='test', status='in_progress')
        s.complete()
        self.assertEqual(s.status, 'completed')
        self.assertIsNotNone(s.completed_at)

    def test_fail_sets_status_and_timestamp(self):
        s = Session(session_id='abc', description='test', status='in_progress')
        s.fail()
        self.assertEqual(s.status, 'failed')
        self.assertIsNotNone(s.completed_at)

    def test_seconds_since_completion_returns_none_while_running(self):
        s = Session(session_id='abc', description='test', status='in_progress')
        self.assertIsNone(s.seconds_since_completion)

    def test_seconds_since_completion_after_complete(self):
        s = Session(session_id='abc', description='test', status='in_progress')
        s.complete()
        age = s.seconds_since_completion
        self.assertIsNotNone(age)
        self.assertGreaterEqual(age, 0.0)

    def test_elapsed_grows_over_time(self):
        s = Session(session_id='abc', description='test', status='in_progress')
        time.sleep(0.02)
        self.assertGreater(s.elapsed, 0.0)


# ---------------------------------------------------------------------------
# ChatSessionManager – lifecycle
# ---------------------------------------------------------------------------

class TestChatSessionManagerLifecycle(unittest.TestCase):
    def setUp(self):
        self.mgr = ChatSessionManager(dedup_window_seconds=120)

    def test_start_session_returns_unique_ids(self):
        id1 = self.mgr.start_session('task one')
        id2 = self.mgr.start_session('task two')
        self.assertNotEqual(id1, id2)

    def test_active_session_count_increments(self):
        self.assertEqual(self.mgr.active_session_count(), 0)
        id1 = self.mgr.start_session('task')
        self.assertEqual(self.mgr.active_session_count(), 1)
        id2 = self.mgr.start_session('task2')
        self.assertEqual(self.mgr.active_session_count(), 2)
        self.mgr.complete_session(id1)
        self.assertEqual(self.mgr.active_session_count(), 1)
        self.mgr.complete_session(id2)
        self.assertEqual(self.mgr.active_session_count(), 0)

    def test_complete_session_archives_to_history(self):
        sid = self.mgr.start_session('my task')
        self.mgr.complete_session(sid)
        history = self.mgr.session_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].status, 'completed')

    def test_fail_session_archives_to_history(self):
        sid = self.mgr.start_session('bad task')
        self.mgr.fail_session(sid)
        history = self.mgr.session_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].status, 'failed')

    def test_complete_unknown_session_returns_false(self):
        result = self.mgr.complete_session('non-existent-id')
        self.assertFalse(result)

    def test_fail_unknown_session_returns_false(self):
        result = self.mgr.fail_session('non-existent-id')
        self.assertFalse(result)

    def test_get_session_returns_live_session(self):
        sid = self.mgr.start_session('live')
        session = self.mgr.get_session(sid)
        self.assertIsNotNone(session)
        self.assertEqual(session.status, 'in_progress')

    def test_get_session_returns_none_after_completion(self):
        sid = self.mgr.start_session('done')
        self.mgr.complete_session(sid)
        self.assertIsNone(self.mgr.get_session(sid))

    def test_history_capped_at_max_history(self):
        mgr = ChatSessionManager(max_history=3)
        for i in range(5):
            sid = mgr.start_session(f'task {i}')
            mgr.complete_session(sid)
        self.assertLessEqual(len(mgr.session_history()), 3)


# ---------------------------------------------------------------------------
# ChatSessionManager – deduplication
# ---------------------------------------------------------------------------

class TestChatSessionManagerDeduplication(unittest.TestCase):
    def setUp(self):
        self.mgr = ChatSessionManager(dedup_window_seconds=120)

    # -- suppress when in_progress ------------------------------------------

    def test_suppress_confirmation_while_session_in_progress(self):
        self.mgr.start_session('active task')
        self.assertTrue(self.mgr.should_suppress('yes'))

    def test_suppress_copilot_phrase_while_session_in_progress(self):
        self.mgr.start_session('active task')
        self.assertTrue(
            self.mgr.should_suppress('@Copilot Accepted Confirmation: Confirm agent session')
        )

    def test_suppress_merge_done_while_session_in_progress(self):
        self.mgr.start_session('active task')
        self.assertTrue(self.mgr.should_suppress('merge done'))

    # -- suppress within dedup window after completion ----------------------

    def test_suppress_confirmation_within_dedup_window(self):
        sid = self.mgr.start_session('just completed')
        self.mgr.complete_session(sid)
        # Completed < 1 s ago; dedup window is 120 s → suppress.
        self.assertTrue(self.mgr.should_suppress('confirmed'))

    def test_allow_confirmation_outside_dedup_window(self):
        mgr = ChatSessionManager(dedup_window_seconds=0)
        sid = mgr.start_session('old task')
        mgr.complete_session(sid)
        # Window is 0 → don't suppress (time elapsed ≥ 0 always satisfies age > 0).
        # Allow because age (even tiny) >= dedup_window (0 only when equal, but
        # we check age <= window, and age > 0 when window == 0).
        self.assertFalse(mgr.should_suppress('ok'))

    # -- never suppress substantive requests --------------------------------

    def test_never_suppress_substantive_request_during_active_session(self):
        self.mgr.start_session('active')
        self.assertFalse(
            self.mgr.should_suppress('open Brave and search for Python tutorials')
        )

    def test_never_suppress_substantive_request_after_completion(self):
        sid = self.mgr.start_session('done')
        self.mgr.complete_session(sid)
        self.assertFalse(
            self.mgr.should_suppress('add dark mode to the settings panel')
        )

    # -- allow when no sessions exist and no history ------------------------

    def test_allow_when_no_sessions_at_all(self):
        self.assertFalse(self.mgr.should_suppress('yes'))

    # -- suppression_reason -------------------------------------------------

    def test_suppression_reason_in_progress(self):
        self.mgr.start_session('some task')
        reason = self.mgr.suppression_reason('confirmed')
        self.assertIn('already in progress', reason)

    def test_suppression_reason_after_completion(self):
        sid = self.mgr.start_session('some task')
        self.mgr.complete_session(sid)
        reason = self.mgr.suppression_reason('ok')
        self.assertIn('completed', reason)

    def test_suppression_reason_empty_for_substantive_request(self):
        self.mgr.start_session('active')
        reason = self.mgr.suppression_reason('open Brave and search for Python')
        self.assertEqual(reason, '')

    # -- static proxy -------------------------------------------------------

    def test_static_is_confirmation_phrase_proxy(self):
        self.assertTrue(ChatSessionManager.is_confirmation_phrase('yes'))
        self.assertFalse(ChatSessionManager.is_confirmation_phrase('open Brave'))


if __name__ == '__main__':
    unittest.main()
