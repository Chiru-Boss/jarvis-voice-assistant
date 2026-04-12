"""Tests for core/swipe_keyboard.py – Air Swipe Typing Keyboard.

Covers:
- QWERTY layout completeness and geometry
- Swipe lifecycle (begin / update / end)
- Key sequence collection
- Hold-repeat for double letters
- Live overlay state (path, raw_word, suggestion, highlighted_key)
- Auto-correction via Levenshtein distance
- DEL key handling
- Empty swipe handling
"""

from __future__ import annotations

import sys
import os
import time
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.swipe_keyboard import (
    SwipeKeyboard,
    SwipeOverlayState,
    QWERTY_ROWS,
    _levenshtein,
    _load_dictionary,
)


# ---------------------------------------------------------------------------
# Levenshtein helpers
# ---------------------------------------------------------------------------

class TestLevenshtein(unittest.TestCase):
    """Unit tests for the pure-Python Levenshtein implementation."""

    def test_identical_strings(self):
        self.assertEqual(_levenshtein('hello', 'hello'), 0)

    def test_empty_strings(self):
        self.assertEqual(_levenshtein('', ''), 0)

    def test_one_empty(self):
        self.assertEqual(_levenshtein('abc', ''), 3)
        self.assertEqual(_levenshtein('', 'abc'), 3)

    def test_single_substitution(self):
        self.assertEqual(_levenshtein('cat', 'bat'), 1)

    def test_single_insertion(self):
        self.assertEqual(_levenshtein('helo', 'hello'), 1)

    def test_single_deletion(self):
        self.assertEqual(_levenshtein('helllo', 'hello'), 1)

    def test_transposition(self):
        # 'wrold' → 'world' requires 2 edits (swap r/o)
        self.assertGreater(_levenshtein('wrold', 'world'), 0)

    def test_completely_different(self):
        self.assertGreater(_levenshtein('abc', 'xyz'), 0)


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------

class TestDictionaryLoader(unittest.TestCase):
    """Verify the word list loads and contains key words."""

    def setUp(self):
        self.words = _load_dictionary()

    def test_returns_list(self):
        self.assertIsInstance(self.words, list)

    def test_non_empty(self):
        self.assertGreater(len(self.words), 0)

    def test_all_lowercase(self):
        for word in self.words[:100]:
            self.assertEqual(word, word.lower(), f"Word not lowercase: {word!r}")

    def test_contains_hello(self):
        self.assertIn('hello', self.words)


# ---------------------------------------------------------------------------
# QWERTY layout
# ---------------------------------------------------------------------------

class TestQWERTYLayout(unittest.TestCase):
    """Verify the keyboard layout is correctly defined."""

    def test_all_rows_present(self):
        self.assertEqual(len(QWERTY_ROWS), 4)

    def test_top_row_letters(self):
        top_row = QWERTY_ROWS[0]
        self.assertEqual(top_row, ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'])

    def test_home_row_letters(self):
        home_row = QWERTY_ROWS[1]
        self.assertEqual(home_row, ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'])

    def test_bottom_row_letters(self):
        bottom_row = QWERTY_ROWS[2]
        self.assertEqual(bottom_row, ['Z', 'X', 'C', 'V', 'B', 'N', 'M'])

    def test_special_keys_row(self):
        special_row = QWERTY_ROWS[3]
        self.assertIn('SPACE', special_row)
        self.assertIn('DEL', special_row)
        self.assertIn('ENTER', special_row)

    def test_all_english_letters_present(self):
        all_keys = [key for row in QWERTY_ROWS for key in row]
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.assertIn(letter, all_keys, f"Missing letter: {letter}")


# ---------------------------------------------------------------------------
# KeyCell geometry
# ---------------------------------------------------------------------------

class TestKeyCellGeometry(unittest.TestCase):
    """Verify generated key cells have valid normalised positions."""

    def setUp(self):
        self.kb = SwipeKeyboard()
        self.cells = self.kb._cells

    def test_cell_count(self):
        # Total keys: 10 + 9 + 7 + 3 = 29
        expected = sum(len(row) for row in QWERTY_ROWS)
        self.assertEqual(len(self.cells), expected)

    def test_x_centers_in_range(self):
        for cell in self.cells:
            self.assertGreaterEqual(cell.x_center, 0.0)
            self.assertLessEqual(cell.x_center, 1.0)

    def test_y_centers_in_range(self):
        for cell in self.cells:
            self.assertGreater(cell.y_center, 0.5, "Keys should be in lower half of frame")
            self.assertLessEqual(cell.y_center, 1.0)

    def test_positive_half_extents(self):
        for cell in self.cells:
            self.assertGreater(cell.x_half, 0.0)
            self.assertGreater(cell.y_half, 0.0)

    def test_key_labels_match_qwerty(self):
        all_expected = [key for row in QWERTY_ROWS for key in row]
        cell_labels = [cell.label for cell in self.cells]
        self.assertEqual(sorted(cell_labels), sorted(all_expected))


# ---------------------------------------------------------------------------
# SwipeKeyboard – lifecycle
# ---------------------------------------------------------------------------

class TestSwipeKeyboardLifecycle(unittest.TestCase):
    """Test the begin / update / end swipe lifecycle."""

    def setUp(self):
        self.kb = SwipeKeyboard()

    def test_initial_state(self):
        self.assertFalse(self.kb.is_active)

    def test_begin_swipe_activates(self):
        self.kb.begin_swipe()
        self.assertTrue(self.kb.is_active)

    def test_end_swipe_deactivates(self):
        self.kb.begin_swipe()
        self.kb.end_swipe()
        self.assertFalse(self.kb.is_active)

    def test_end_swipe_returns_string(self):
        self.kb.begin_swipe()
        result = self.kb.end_swipe()
        self.assertIsInstance(result, str)

    def test_empty_swipe_returns_empty_string(self):
        self.kb.begin_swipe()
        result = self.kb.end_swipe()
        self.assertEqual(result, '')

    def test_update_before_begin_is_safe(self):
        # Should not raise even when inactive
        self.kb.update_swipe(0.5, 0.7)  # no-op

    def test_begin_clears_previous_state(self):
        # First swipe
        self.kb.begin_swipe()
        self._swipe_word('HI')
        self.kb.end_swipe()
        # Second swipe: state must be fresh
        self.kb.begin_swipe()
        state = self.kb.get_overlay_state()
        self.assertEqual(state.raw_word, '')
        self.assertEqual(state.path, [])

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _swipe_word(self, letters: str) -> None:
        """Simulate swiping over the given letters sequentially."""
        for letter in letters.upper():
            cell = self._find_cell(letter)
            if cell:
                self.kb.update_swipe(cell.x_center, cell.y_center)

    def _find_cell(self, label: str):
        for cell in self.kb._cells:
            if cell.label == label:
                return cell
        return None


# ---------------------------------------------------------------------------
# SwipeKeyboard – key collection
# ---------------------------------------------------------------------------

class TestSwipeKeyCollection(unittest.TestCase):
    """Test that swiping over keys collects the right sequence."""

    def setUp(self):
        self.kb = SwipeKeyboard()
        self.kb.begin_swipe()

    def _find_cell(self, label: str):
        for cell in self.kb._cells:
            if cell.label == label:
                return cell
        return None

    def _swipe_over(self, *labels: str) -> None:
        for label in labels:
            cell = self._find_cell(label)
            if cell:
                self.kb.update_swipe(cell.x_center, cell.y_center)

    def test_single_key_collected(self):
        self._swipe_over('H')
        state = self.kb.get_overlay_state()
        self.assertEqual(state.raw_word, 'h')

    def test_multiple_keys_collected(self):
        self._swipe_over('H', 'E', 'L')
        state = self.kb.get_overlay_state()
        self.assertIn('h', state.raw_word)
        self.assertIn('e', state.raw_word)
        self.assertIn('l', state.raw_word)

    def test_duplicate_adjacent_key_not_repeated(self):
        # Moving over the same key twice quickly should only add it once
        cell = self._find_cell('A')
        if cell:
            self.kb.update_swipe(cell.x_center, cell.y_center)
            self.kb.update_swipe(cell.x_center, cell.y_center)
        state = self.kb.get_overlay_state()
        self.assertEqual(state.raw_word, 'a')

    def test_del_removes_last_key(self):
        self._swipe_over('H', 'E', 'L')
        self._swipe_over('DEL')
        state = self.kb.get_overlay_state()
        self.assertEqual(state.raw_word, 'he')

    def test_space_does_not_add_to_raw_word(self):
        self._swipe_over('H', 'I', 'SPACE')
        state = self.kb.get_overlay_state()
        self.assertNotIn(' ', state.raw_word)

    def test_enter_does_not_add_to_raw_word(self):
        self._swipe_over('H', 'I', 'ENTER')
        state = self.kb.get_overlay_state()
        self.assertEqual(state.raw_word, 'hi')

    def test_path_accumulated(self):
        self._swipe_over('H', 'E', 'L', 'L', 'O')
        state = self.kb.get_overlay_state()
        self.assertGreater(len(state.path), 0)

    def test_path_length_capped(self):
        # Path is capped at 50 points in get_overlay_state
        for _ in range(100):
            self.kb.update_swipe(0.5, 0.75)
        state = self.kb.get_overlay_state()
        self.assertLessEqual(len(state.path), 50)


# ---------------------------------------------------------------------------
# SwipeKeyboard – hold-repeat for double letters
# ---------------------------------------------------------------------------

class TestHoldRepeat(unittest.TestCase):
    """Test that dwelling on a key triggers hold-repeat (double letters)."""

    def setUp(self):
        # Use very short thresholds so tests don't actually sleep long
        self.kb = SwipeKeyboard(hold_threshold=0.05, hold_interval=0.03)
        self.kb.begin_swipe()

    def _find_cell(self, label: str):
        for cell in self.kb._cells:
            if cell.label == label:
                return cell
        return None

    def test_hold_repeat_adds_extra_letter(self):
        cell = self._find_cell('L')
        if cell is None:
            self.skipTest('L key not found in keyboard layout')

        # First touch – adds 'L' once
        self.kb.update_swipe(cell.x_center, cell.y_center)
        time.sleep(0.06)  # Wait past hold_threshold
        # Second touch on same key – should trigger repeat
        self.kb.update_swipe(cell.x_center, cell.y_center)

        state = self.kb.get_overlay_state()
        self.assertGreaterEqual(state.raw_word.count('l'), 2,
                                f"Expected double-l, got raw_word={state.raw_word!r}")

    def test_no_hold_repeat_without_wait(self):
        cell = self._find_cell('L')
        if cell is None:
            self.skipTest('L key not found in keyboard layout')

        # Touch L twice quickly (no sleep)
        self.kb.update_swipe(cell.x_center, cell.y_center)
        self.kb.update_swipe(cell.x_center, cell.y_center)

        state = self.kb.get_overlay_state()
        self.assertEqual(state.raw_word.count('l'), 1,
                         f"Expected single-l, got raw_word={state.raw_word!r}")


# ---------------------------------------------------------------------------
# SwipeKeyboard – auto-correction
# ---------------------------------------------------------------------------

class TestAutoCorrection(unittest.TestCase):
    """Test Levenshtein-based auto-correction."""

    def setUp(self):
        self.kb = SwipeKeyboard()

    def _find_cell(self, label: str):
        for cell in self.kb._cells:
            if cell.label == label:
                return cell
        return None

    def _swipe_word(self, letters: str) -> str:
        self.kb.begin_swipe()
        for letter in letters.upper():
            cell = self._find_cell(letter)
            if cell:
                self.kb.update_swipe(cell.x_center, cell.y_center)
        return self.kb.end_swipe()

    def test_hlp_corrects_to_help(self):
        # Swiping H→L→P gives raw 'hlp'; only 'help' is at distance 1 in the
        # dictionary, so auto-correction should return 'help'.
        result = self._swipe_word('HLP')
        self.assertEqual(result, 'help')

    def test_wrd_corrects_to_word(self):
        # Swiping W→R→D gives raw 'wrd'; closest match is 'word' (distance 1).
        result = self._swipe_word('WRD')
        self.assertEqual(result, 'word')

    def test_exact_word_returns_itself(self):
        # HELP has no duplicate adjacent letters so raw_word = 'help' exactly.
        result = self._swipe_word('HELP')
        self.assertEqual(result, 'help')

    def test_suggestion_updates_live(self):
        self.kb.begin_swipe()
        for letter in 'HELL':
            cell = self._find_cell(letter)
            if cell:
                self.kb.update_swipe(cell.x_center, cell.y_center)
        state = self.kb.get_overlay_state()
        # Suggestion should already be populated during swipe
        self.assertIsInstance(state.suggestion, str)
        self.assertGreater(len(state.suggestion), 0,
                           "Live suggestion should be non-empty after swiping HELL")

    def test_very_short_word_not_corrected(self):
        # Single-char words below min length are returned as-is
        result = self._swipe_word('A')
        self.assertEqual(result, 'a')

    def test_no_correction_needed(self):
        result = self._swipe_word('THE')
        self.assertIn(result, ('the',))  # 'the' is in the word list

    def test_end_swipe_uses_suggestion(self):
        """end_swipe() should prefer the suggestion over the raw word."""
        self.kb.begin_swipe()
        for letter in 'HELO':
            cell = self._find_cell(letter)
            if cell:
                self.kb.update_swipe(cell.x_center, cell.y_center)
        state = self.kb.get_overlay_state()
        result = self.kb.end_swipe()
        # If a suggestion was generated, result must equal it
        if state.suggestion:
            self.assertEqual(result, state.suggestion)


# ---------------------------------------------------------------------------
# SwipeKeyboard – overlay state
# ---------------------------------------------------------------------------

class TestOverlayState(unittest.TestCase):
    """Verify get_overlay_state() returns complete, correct data."""

    def setUp(self):
        self.kb = SwipeKeyboard()

    def test_inactive_state(self):
        state = self.kb.get_overlay_state()
        self.assertIsInstance(state, SwipeOverlayState)
        self.assertFalse(state.is_active)
        self.assertIsInstance(state.key_cells, list)
        self.assertGreater(len(state.key_cells), 0)

    def test_active_state_after_begin(self):
        self.kb.begin_swipe()
        state = self.kb.get_overlay_state()
        self.assertTrue(state.is_active)

    def test_highlighted_key_when_active(self):
        self.kb.begin_swipe()
        # Move to 'H' key
        h_cell = next((c for c in self.kb._cells if c.label == 'H'), None)
        if h_cell:
            self.kb.update_swipe(h_cell.x_center, h_cell.y_center)
            state = self.kb.get_overlay_state()
            self.assertEqual(state.highlighted_key, 'H')

    def test_no_highlighted_key_when_inactive(self):
        state = self.kb.get_overlay_state()
        self.assertIsNone(state.highlighted_key)

    def test_completed_word_after_end(self):
        self.kb.begin_swipe()
        h_cell = next((c for c in self.kb._cells if c.label == 'H'), None)
        if h_cell:
            self.kb.update_swipe(h_cell.x_center, h_cell.y_center)
        word = self.kb.end_swipe()
        state = self.kb.get_overlay_state()
        self.assertEqual(state.completed_word, word)

    def test_state_fields_present(self):
        state = self.kb.get_overlay_state()
        self.assertTrue(hasattr(state, 'is_active'))
        self.assertTrue(hasattr(state, 'key_cells'))
        self.assertTrue(hasattr(state, 'highlighted_key'))
        self.assertTrue(hasattr(state, 'path'))
        self.assertTrue(hasattr(state, 'raw_word'))
        self.assertTrue(hasattr(state, 'suggestion'))
        self.assertTrue(hasattr(state, 'completed_word'))


# ---------------------------------------------------------------------------
# SwipeKeyboard – nearest key hit-testing
# ---------------------------------------------------------------------------

class TestNearestKey(unittest.TestCase):
    """Test the _nearest_key() hit-testing method."""

    def setUp(self):
        self.kb = SwipeKeyboard()

    def test_hit_key_directly(self):
        # Place finger exactly on 'A' key centre
        a_cell = next((c for c in self.kb._cells if c.label == 'A'), None)
        self.assertIsNotNone(a_cell)
        result = self.kb._nearest_key(a_cell.x_center, a_cell.y_center)
        self.assertEqual(result, 'A')

    def test_outside_keyboard_zone_returns_none(self):
        # Very top of frame is outside keyboard area
        result = self.kb._nearest_key(0.5, 0.1)
        self.assertIsNone(result)

    def test_off_left_edge_returns_none(self):
        result = self.kb._nearest_key(-0.5, 0.78)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
