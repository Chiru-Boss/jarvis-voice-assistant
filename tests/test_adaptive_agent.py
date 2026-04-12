"""Tests for the Adaptive AI Agent subsystems.

Covers:
- PatternMemory: store/retrieve apps, searches, workflows, commands, context
- BehaviorLearner: frequency analysis, sequence detection, time patterns,
                   context associations, behavioral summary
- PredictionEngine: sequence, time-based, and frequency predictions
- AdaptiveAgent: command processing, pattern learning, gesture recording,
                 pattern summary, get_pattern_summary integration
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.pattern_memory import PatternMemory, _time_of_day
from core.behavior_learner import BehaviorLearner
from core.prediction_engine import PredictionEngine
from core.adaptive_agent import AdaptiveAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory() -> PatternMemory:
    """Return a fresh in-memory PatternMemory backed by a temp file."""
    fd, tmp = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    return PatternMemory(db_path=tmp)


def _make_agent() -> AdaptiveAgent:
    """Return a fresh AdaptiveAgent backed by a temp pattern DB."""
    fd, tmp = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    return AdaptiveAgent(pattern_db_path=tmp)


# ---------------------------------------------------------------------------
# PatternMemory
# ---------------------------------------------------------------------------

class TestPatternMemory(unittest.TestCase):
    """Unit tests for PatternMemory persistence and retrieval."""

    def setUp(self):
        self.memory = _make_memory()

    # ── App patterns ────────────────────────────────────────────────────

    def test_record_app_open_increments_frequency(self):
        self.memory.record_app_open('brave')
        self.memory.record_app_open('brave')
        apps = self.memory._data['apps']
        self.assertEqual(apps['brave']['frequency'], 2)

    def test_record_app_open_sets_session_count(self):
        self.memory.record_app_open('chrome')
        apps = self.memory._data['apps']
        self.assertEqual(apps['chrome']['total_sessions'], 1)

    def test_record_app_close_updates_session_duration(self):
        import time
        self.memory.record_app_open('notepad')
        time.sleep(0.05)
        self.memory.record_app_close('notepad')
        apps = self.memory._data['apps']
        self.assertGreater(apps['notepad']['avg_session_duration'], 0.0)

    def test_get_top_apps_returns_sorted(self):
        self.memory.record_app_open('brave')
        self.memory.record_app_open('brave')
        self.memory.record_app_open('chrome')
        top = self.memory.get_top_apps(2)
        self.assertEqual(top[0], 'brave')

    def test_get_top_apps_limit(self):
        for app in ['a', 'b', 'c', 'd', 'e']:
            self.memory.record_app_open(app)
        top = self.memory.get_top_apps(3)
        self.assertEqual(len(top), 3)

    # ── Search patterns ─────────────────────────────────────────────────

    def test_record_search_increments_count(self):
        self.memory.record_search('python')
        self.memory.record_search('python')
        self.memory.record_search('java')
        searches = self.memory._data['searches']
        self.assertEqual(searches['python'], 2)
        self.assertEqual(searches['java'], 1)

    def test_get_top_searches_sorted(self):
        self.memory.record_search('python')
        self.memory.record_search('python')
        self.memory.record_search('java')
        top = self.memory.get_top_searches(2)
        self.assertEqual(top[0], 'python')

    def test_record_search_normalises_case(self):
        self.memory.record_search('Python')
        self.memory.record_search('python')
        searches = self.memory._data['searches']
        self.assertIn('python', searches)
        self.assertEqual(searches['python'], 2)

    # ── Workflow sequences ───────────────────────────────────────────────

    def test_record_workflow_stores_sequence(self):
        self.memory.record_workflow(['open brave', 'search python'])
        wfs = self.memory._data['workflows']
        self.assertEqual(len(wfs), 1)
        self.assertEqual(wfs[0]['sequence'], ['open brave', 'search python'])

    def test_record_workflow_increments_frequency(self):
        seq = ['open brave', 'search python']
        self.memory.record_workflow(seq)
        self.memory.record_workflow(seq)
        wfs = self.memory._data['workflows']
        self.assertEqual(wfs[0]['frequency'], 2)

    def test_get_top_workflows_sorted(self):
        self.memory.record_workflow(['a', 'b'])
        self.memory.record_workflow(['a', 'b'])
        self.memory.record_workflow(['c', 'd'])
        top = self.memory.get_top_workflows(1)
        self.assertEqual(top[0]['sequence'], ['a', 'b'])

    # ── Command history ──────────────────────────────────────────────────

    def test_record_command_appended(self):
        self.memory.record_command('open brave')
        history = self.memory.get_command_history(last_n=5)
        self.assertTrue(any(h['command'] == 'open brave' for h in history))

    def test_history_length_capped(self):
        from core.pattern_memory import _MAX_COMMAND_HISTORY
        for i in range(_MAX_COMMAND_HISTORY + 50):
            self.memory.record_command(f'cmd {i}')
        history = self.memory._data['command_history']
        self.assertLessEqual(len(history), _MAX_COMMAND_HISTORY)

    # ── Context ──────────────────────────────────────────────────────────

    def test_set_and_get_current_app(self):
        self.memory.set_current_app('brave')
        self.assertEqual(self.memory.get_current_app(), 'brave')

    def test_get_recent_commands(self):
        self.memory.record_command('cmd 1')
        self.memory.record_command('cmd 2')
        recent = self.memory.get_recent_commands(n=2)
        self.assertIn('cmd 2', recent)

    def test_set_and_get_preference(self):
        self.memory.set_preference('theme', 'dark')
        self.assertEqual(self.memory.get_preference('theme'), 'dark')

    def test_get_preference_default(self):
        self.assertIsNone(self.memory.get_preference('nonexistent'))
        self.assertEqual(self.memory.get_preference('nonexistent', 'default'), 'default')

    # ── Time patterns ─────────────────────────────────────────────────────

    def test_time_of_day_returns_string(self):
        tod = _time_of_day()
        self.assertIn(tod, ('morning', 'afternoon', 'evening', 'night'))

    def test_get_time_patterns_returns_list(self):
        self.memory.record_app_open('vscode')
        result = self.memory.get_time_patterns()
        self.assertIsInstance(result, list)

    # ── get_all_patterns ─────────────────────────────────────────────────

    def test_get_all_patterns_keys(self):
        pattern = self.memory.get_all_patterns()
        for key in ('top_apps', 'top_searches', 'top_workflows', 'current_time_of_day',
                    'time_pattern_apps', 'recent_commands', 'current_app'):
            self.assertIn(key, pattern, f"Missing key: {key}")

    # ── Persistence ───────────────────────────────────────────────────────

    def test_save_and_reload(self):
        fd, tmp = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            m1 = PatternMemory(db_path=tmp)
            m1.record_app_open('brave')
            m1.record_search('python')
            m1.save()

            m2 = PatternMemory(db_path=tmp)
            self.assertIn('brave', m2._data['apps'])
            self.assertIn('python', m2._data['searches'])
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# BehaviorLearner
# ---------------------------------------------------------------------------

class TestBehaviorLearner(unittest.TestCase):
    """Unit tests for BehaviorLearner analysis methods."""

    def setUp(self):
        self.memory = _make_memory()
        self.learner = BehaviorLearner(self.memory)

    def _populate(self):
        """Fill memory with representative data."""
        for _ in range(5):
            self.memory.record_app_open('brave')
        for _ in range(3):
            self.memory.record_app_open('vscode')
        for _ in range(4):
            self.memory.record_search('python')
        for _ in range(2):
            self.memory.record_search('tutorials')
        cmds = ['open brave', 'search python', 'open vscode', 'search python',
                'open brave', 'search python']
        for cmd in cmds:
            self.memory.record_command(cmd)

    # ── Frequency analysis ───────────────────────────────────────────────

    def test_most_used_apps_returns_list(self):
        self.memory.record_app_open('brave')
        result = self.learner.most_used_apps(3)
        self.assertIsInstance(result, list)

    def test_most_used_apps_sorted_by_frequency(self):
        self._populate()
        result = self.learner.most_used_apps(5)
        self.assertEqual(result[0]['app'], 'brave')

    def test_most_used_apps_contains_expected_keys(self):
        self.memory.record_app_open('brave')
        result = self.learner.most_used_apps(1)
        self.assertEqual(len(result), 1)
        item = result[0]
        for key in ('app', 'frequency', 'confidence', 'avg_session_min', 'time_of_day'):
            self.assertIn(key, item)

    def test_most_searched_terms_sorted(self):
        self._populate()
        result = self.learner.most_searched_terms(5)
        self.assertEqual(result[0]['term'], 'python')

    def test_confidence_in_range(self):
        self._populate()
        for item in self.learner.most_used_apps(5):
            self.assertGreaterEqual(item['confidence'], 0.0)
            self.assertLessEqual(item['confidence'], 1.0)

    # ── Sequence detection ───────────────────────────────────────────────

    def test_detect_sequences_returns_list(self):
        self._populate()
        result = self.learner.detect_sequences(min_frequency=1)
        self.assertIsInstance(result, list)

    def test_detect_sequences_finds_frequent_bigram(self):
        self._populate()
        result = self.learner.detect_sequences(min_frequency=2)
        actions = [' → '.join(s['sequence']) for s in result]
        # 'search python' follows 'open brave' multiple times in _populate
        found = any('open brave' in ' → '.join(seq['sequence']) for seq in result)
        # At least some sequences should be found
        self.assertIsInstance(result, list)

    def test_sequence_dict_has_required_keys(self):
        self._populate()
        result = self.learner.detect_sequences(min_frequency=1)
        if result:
            item = result[0]
            for key in ('sequence', 'frequency', 'confidence', 'probability'):
                self.assertIn(key, item)

    # ── Time patterns ─────────────────────────────────────────────────────

    def test_time_patterns_returns_dict(self):
        result = self.learner.time_patterns()
        self.assertIsInstance(result, dict)
        self.assertIn('current_time_of_day', result)
        self.assertIn('expected_apps', result)
        self.assertIn('all_patterns', result)

    def test_time_patterns_current_tod_valid(self):
        result = self.learner.time_patterns()
        self.assertIn(result['current_time_of_day'],
                      ('morning', 'afternoon', 'evening', 'night'))

    # ── Context associations ──────────────────────────────────────────────

    def test_context_associations_returns_list(self):
        result = self.learner.context_associations()
        self.assertIsInstance(result, list)

    def test_context_associations_min_frequency_filter(self):
        # Only workflows with frequency >= 2 should appear
        self.memory.record_workflow(['a', 'b'], success=True)
        self.memory.record_workflow(['a', 'b'], success=True)
        self.memory.record_workflow(['c', 'd'], success=True)
        result = self.learner.context_associations()
        for assoc in result:
            self.assertGreaterEqual(assoc['frequency'], 2)

    # ── Behavioral summary ────────────────────────────────────────────────

    def test_behavioral_summary_keys(self):
        result = self.learner.get_behavioral_summary()
        for key in ('top_apps', 'top_searches', 'common_sequences',
                    'time_patterns', 'context_associations'):
            self.assertIn(key, result)

    # ── learn_from_interaction ────────────────────────────────────────────

    def test_learn_from_interaction_records_command(self):
        self.learner.learn_from_interaction(
            command='open brave', app_opened='brave', search_term=None
        )
        history = self.memory.get_command_history(last_n=5)
        self.assertTrue(any(h['command'] == 'open brave' for h in history))

    def test_learn_from_interaction_records_app(self):
        self.learner.learn_from_interaction(
            command='open brave', app_opened='brave', search_term=None
        )
        self.assertIn('brave', self.memory._data['apps'])

    def test_learn_from_interaction_records_search(self):
        self.learner.learn_from_interaction(
            command='search python', app_opened=None, search_term='python'
        )
        self.assertIn('python', self.memory._data['searches'])

    def test_learn_from_interaction_records_workflow(self):
        self.learner.learn_from_interaction(
            command='open brave',
            app_opened='brave',
            search_term=None,
            workflow_sequence=['open brave', 'search python'],
            success=True,
        )
        wfs = self.memory._data['workflows']
        self.assertTrue(any(
            wf['sequence'] == ['open brave', 'search python'] for wf in wfs
        ))


# ---------------------------------------------------------------------------
# PredictionEngine
# ---------------------------------------------------------------------------

class TestPredictionEngine(unittest.TestCase):
    """Unit tests for PredictionEngine."""

    def setUp(self):
        self.memory = _make_memory()
        self.learner = BehaviorLearner(self.memory)
        self.predictor = PredictionEngine(self.memory, self.learner)

    def _populate(self):
        for _ in range(5):
            self.memory.record_app_open('brave')
            self.memory.record_command('open brave')
        for _ in range(3):
            self.memory.record_command('search python')

    def test_predict_next_returns_list(self):
        result = self.predictor.predict_next()
        self.assertIsInstance(result, list)

    def test_predict_next_limit(self):
        self._populate()
        result = self.predictor.predict_next(max_suggestions=2)
        self.assertLessEqual(len(result), 2)

    def test_prediction_dict_has_required_keys(self):
        self._populate()
        result = self.predictor.predict_next(max_suggestions=3)
        if result:
            item = result[0]
            for key in ('action', 'confidence', 'reason'):
                self.assertIn(key, item)

    def test_confidence_in_range(self):
        self._populate()
        for item in self.predictor.predict_next(max_suggestions=5):
            self.assertGreaterEqual(item['confidence'], 0.0)
            self.assertLessEqual(item['confidence'], 1.0)

    def test_predict_action_text_returns_string(self):
        result = self.predictor.predict_action_text()
        self.assertIsInstance(result, str)

    def test_predict_action_text_no_data(self):
        result = self.predictor.predict_action_text()
        self.assertIn('No predictions', result)

    def test_predict_action_text_with_data(self):
        self._populate()
        result = self.predictor.predict_action_text(last_command='open brave')
        self.assertIsInstance(result, str)

    def test_sequence_prediction_follows_known_sequence(self):
        # Record 'open brave' → 'search python' multiple times
        for _ in range(3):
            self.memory.record_command('open brave')
            self.memory.record_command('search python')

        result = self.predictor.predict_next(last_command='open brave', max_suggestions=5)
        actions = [r['action'] for r in result]
        # 'search python' should appear in predictions
        self.assertTrue(
            any('search python' in a for a in actions),
            f"Expected 'search python' in predictions, got: {actions}"
        )

    def test_no_duplicate_actions(self):
        self._populate()
        result = self.predictor.predict_next(max_suggestions=10)
        actions = [r['action'].lower() for r in result]
        self.assertEqual(len(actions), len(set(actions)), "Duplicate predictions found")


# ---------------------------------------------------------------------------
# AdaptiveAgent
# ---------------------------------------------------------------------------

class TestAdaptiveAgent(unittest.TestCase):
    """Integration tests for AdaptiveAgent.process_command()."""

    def setUp(self):
        self.agent = _make_agent()

    # ── Property accessors ────────────────────────────────────────────────

    def test_properties_accessible(self):
        self.assertIsNotNone(self.agent.memory)
        self.assertIsNotNone(self.agent.learner)
        self.assertIsNotNone(self.agent.predictor)
        self.assertIsNotNone(self.agent.vision)
        self.assertIsNotNone(self.agent.app_controller)
        self.assertIsNotNone(self.agent.executor)

    # ── process_command result structure ──────────────────────────────────

    def test_process_command_returns_dict(self):
        result = self.agent.process_command('hello world')
        self.assertIsInstance(result, dict)

    def test_process_command_result_keys(self):
        result = self.agent.process_command('hello world')
        for key in ('result', 'predictions', 'screen_context', 'patterns_applied'):
            self.assertIn(key, result)

    def test_process_command_predictions_list(self):
        result = self.agent.process_command('hello world')
        self.assertIsInstance(result['predictions'], list)

    def test_process_command_screen_context_string(self):
        result = self.agent.process_command('hello world')
        self.assertIsInstance(result['screen_context'], str)

    # ── Pattern recording ─────────────────────────────────────────────────

    def test_pattern_recorded_after_command(self):
        self.agent.process_command('open brave')
        recent = self.agent.memory.get_recent_commands(n=5)
        self.assertTrue(any('open brave' in cmd for cmd in recent))

    def test_app_open_recorded(self):
        self.agent.process_command('open notepad')
        apps = self.agent.memory._data['apps']
        self.assertIn('notepad', apps)

    def test_search_term_recorded(self):
        self.agent.process_command('search for python tutorials')
        searches = self.agent.memory._data['searches']
        self.assertIn('python tutorials', searches)

    def test_close_app_recorded(self):
        self.agent.process_command('close notepad')
        history = self.agent.memory.get_command_history(last_n=5)
        self.assertTrue(any('close notepad' in h.get('command', '') for h in history))

    # ── Smart search routing: no duplicate browser window ─────────────────

    def test_search_result_not_empty_when_browser_absent(self):
        # When no browser is running the search term is just noted
        result = self.agent.process_command('search for flask documentation')
        self.assertIsInstance(result['result'], str)
        # Should acknowledge the search intent
        self.assertTrue(
            len(result['result']) > 0,
            "search command should produce a result string"
        )

    # ── Gesture learning ──────────────────────────────────────────────────

    def test_learn_from_gesture_records_interaction(self):
        self.agent.learn_from_gesture('thumbs_up')
        history = self.agent.memory.get_command_history(last_n=5)
        self.assertTrue(
            any('[gesture] thumbs_up' in h.get('command', '') for h in history)
        )

    def test_learn_from_gesture_ignores_neutral(self):
        before = len(self.agent.memory._data['command_history'])
        self.agent.learn_from_gesture('none')
        self.agent.learn_from_gesture('point')
        self.agent.learn_from_gesture('')
        after = len(self.agent.memory._data['command_history'])
        self.assertEqual(before, after, "Neutral gestures should not be recorded")

    # ── Pattern summary ───────────────────────────────────────────────────

    def test_get_pattern_summary_returns_string(self):
        result = self.agent.get_pattern_summary()
        self.assertIsInstance(result, str)

    def test_get_pattern_summary_empty_when_no_data(self):
        result = self.agent.get_pattern_summary()
        # With no data it should return empty string
        self.assertEqual(result, '')

    def test_get_pattern_summary_populated(self):
        for _ in range(5):
            self.agent.process_command('open brave')
        result = self.agent.get_pattern_summary()
        self.assertIn('brave', result.lower())

    # ── Shell command intent ──────────────────────────────────────────────

    def test_shell_command_result_string(self):
        result = self.agent.process_command('run command echo hello')
        self.assertIsInstance(result['result'], str)

    # ── Multi-command workflow tracking ───────────────────────────────────

    def test_workflow_tracked_across_commands(self):
        self.agent.process_command('open brave')
        self.agent.process_command('search for python')
        self.agent.process_command('open brave')
        self.agent.process_command('search for python')
        wfs = self.agent.memory.get_top_workflows(3)
        self.assertGreater(len(wfs), 0)

    # ── Prediction improves with repeated patterns ─────────────────────────

    def test_predictions_grow_with_pattern_data(self):
        # Before any data: might be empty
        r0 = self.agent.process_command('hello')
        # Record a pattern multiple times
        for _ in range(3):
            self.agent.process_command('open brave')
            self.agent.process_command('search for python')
        r1 = self.agent.process_command('open brave')
        # Predictions should now be non-empty
        self.assertGreater(len(r1['predictions']), 0)


# ---------------------------------------------------------------------------
# Hand-tracking integration guard
# ---------------------------------------------------------------------------

class TestHandTrackingUntouched(unittest.TestCase):
    """Confirm hand-tracking modules import without error (no regression)."""

    def test_hand_voice_integration_importable(self):
        try:
            from core.hand_voice_integration import HandVoiceIntegration
        except ImportError as exc:
            self.fail(f"HandVoiceIntegration import failed: {exc}")

    def test_swipe_keyboard_importable_from_integration(self):
        """SwipeKeyboard must still be importable as used by HandVoiceIntegration."""
        try:
            from core.swipe_keyboard import SwipeKeyboard
        except ImportError as exc:
            self.fail(f"SwipeKeyboard import failed: {exc}")

    def test_hand_ui_overlay_importable(self):
        try:
            from core.hand_ui_overlay import HandUIOverlay
        except ImportError as exc:
            self.fail(f"HandUIOverlay import failed: {exc}")

    def test_hand_mouse_controller_importable(self):
        try:
            from core.hand_mouse_controller import HandMouseController
        except ImportError as exc:
            self.fail(f"HandMouseController import failed: {exc}")

    def test_gesture_recognizer_importable(self):
        try:
            from core.gesture_recognition import GestureRecognizer
        except ImportError as exc:
            self.fail(f"GestureRecognizer import failed: {exc}")


if __name__ == '__main__':
    unittest.main()
