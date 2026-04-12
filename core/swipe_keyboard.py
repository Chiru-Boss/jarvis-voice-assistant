"""Swipe Keyboard – air-gesture swipe typing for JARVIS.

Flow
----
1. Caller detects pinch start  → :meth:`begin_swipe`
2. Every frame while pinching  → :meth:`update_swipe` with index-tip position
3. Caller detects pinch end    → :meth:`end_swipe` → returns corrected word

Features
--------
- QWERTY layout rendered as a keyboard in the lower 40 % of the camera frame.
- Keys are collected into a sequence while the finger moves over the overlay.
- Holding the finger on the same key > 1 s repeats it (e.g. double "l" in "hello").
- On release, Levenshtein distance is used to auto-correct the sequence against
  a bundled English dictionary.
- ``get_overlay_state()`` returns everything needed to draw the live overlay.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyboard layout constants
# ---------------------------------------------------------------------------

QWERTY_ROWS: List[List[str]] = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
    ['SPACE', 'DEL', 'ENTER'],
]

# Normalized (0-1) y-centre for each row inside the keyboard zone
_ROW_Y: List[float] = [0.68, 0.78, 0.87, 0.95]

# Keyboard occupies this x-range (normalised, 0 = left, 1 = right)
_KB_X_LEFT: float  = 0.02
_KB_X_RIGHT: float = 0.98

# Key sizes in normalised units (used for hit-testing)
_KEY_H_HALF: float = 0.045   # half-height of a key cell
# Half-widths are computed per row based on the number of keys

# How long (seconds) a finger must dwell on a key to trigger a hold-repeat
_HOLD_REPEAT_INITIAL: float = 1.0   # first repeat after 1 s
_HOLD_REPEAT_INTERVAL: float = 0.5  # subsequent repeats every 0.5 s

# Maximum Levenshtein distance (relative to word length) to accept a suggestion
_MAX_DISTANCE_RATIO: float = 0.55

# Path to the bundled English word list
_WORDS_FILE = Path(__file__).parent.parent / 'data' / 'english_words.txt'

# ---------------------------------------------------------------------------
# Data classes for overlay state
# ---------------------------------------------------------------------------

@dataclass
class KeyCell:
    """A single key on the virtual keyboard."""
    label: str
    x_center: float   # normalised [0, 1]
    y_center: float   # normalised [0, 1]
    x_half: float     # half-width in normalised units
    y_half: float     # half-height in normalised units


@dataclass
class SwipeOverlayState:
    """Snapshot of swipe-keyboard state for overlay rendering."""
    is_active: bool                           # swipe mode is on
    key_cells: List[KeyCell]                  # all keys (for drawing the board)
    highlighted_key: Optional[str]            # key under finger right now
    path: List[Tuple[float, float]]           # swipe path so far (normalised)
    raw_word: str                             # letters collected so far
    suggestion: str                           # auto-corrected suggestion
    completed_word: str                       # last completed word (after release)


# ---------------------------------------------------------------------------
# Levenshtein distance (pure Python, no external deps)
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between *a* and *b*."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Use two-row DP to save memory
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)

    for i, ca in enumerate(a, 1):
        curr[0] = i
        for j, cb in enumerate(b, 1):
            if ca == cb:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[len(b)]


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------

def _load_dictionary() -> List[str]:
    """Load the English word list, returning a sorted list of lowercase words."""
    if _WORDS_FILE.exists():
        try:
            text = _WORDS_FILE.read_text(encoding='utf-8')
            words = [w.strip().lower() for w in text.splitlines() if w.strip()]
            logger.debug('Loaded %d words from %s.', len(words), _WORDS_FILE)
            return words
        except OSError as exc:
            logger.warning('Could not read word list: %s', exc)

    # Fallback: a small built-in word set for offline / first-boot use
    logger.warning('Using built-in fallback word list (limited).')
    return [
        'hello', 'help', 'held', 'heel', 'here', 'hero', 'high', 'hire', 'hole',
        'home', 'hope', 'horn', 'host', 'hour', 'have', 'head', 'heal', 'hear',
        'heat', 'hide', 'hit', 'hold', 'hop', 'hot', 'how',
        'world', 'word', 'work', 'wore', 'woke', 'won',
        'the', 'that', 'this', 'they', 'them', 'then', 'there', 'their',
        'can', 'car', 'care', 'case', 'cash', 'city', 'call', 'come',
        'you', 'your', 'yet', 'yes',
        'and', 'any', 'are', 'ask', 'also',
        'not', 'now', 'new', 'next',
        'type', 'test', 'team', 'tell', 'time', 'true', 'turn',
        'one', 'open', 'over', 'own',
        'see', 'say', 'set', 'show', 'some', 'soon', 'stop', 'sure',
    ]


# ---------------------------------------------------------------------------
# SwipeKeyboard
# ---------------------------------------------------------------------------

class SwipeKeyboard:
    """Manages swipe-typing gesture state and auto-correction.

    Parameters
    ----------
    hold_threshold : float
        Seconds before a held key triggers its first repeat.
    hold_interval : float
        Seconds between subsequent hold repeats.
    max_distance_ratio : float
        Max Levenshtein distance (as a fraction of word length) accepted as a
        correction.  Set to 0 to disable correction entirely.
    """

    def __init__(
        self,
        hold_threshold: float  = _HOLD_REPEAT_INITIAL,
        hold_interval: float   = _HOLD_REPEAT_INTERVAL,
        max_distance_ratio: float = _MAX_DISTANCE_RATIO,
    ) -> None:
        self._hold_threshold    = hold_threshold
        self._hold_interval     = hold_interval
        self._max_dist_ratio    = max_distance_ratio

        # Build keyboard cell map once
        self._cells: List[KeyCell] = self._build_cells()
        self._dictionary: List[str] = _load_dictionary()

        # Runtime swipe state
        self._is_active: bool = False
        self._path: List[Tuple[float, float]] = []
        self._key_seq: List[str] = []
        self._last_key: Optional[str] = None
        self._last_key_time: float = 0.0
        self._next_repeat_time: float = 0.0
        self._raw_word: str = ''
        self._suggestion: str = ''
        self._completed_word: str = ''

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True while a pinch swipe is in progress."""
        return self._is_active

    def begin_swipe(self) -> None:
        """Call when pinch gesture starts (finger down)."""
        logger.debug('Swipe typing: BEGIN')
        self._is_active = True
        self._path = []
        self._key_seq = []
        self._last_key = None
        self._last_key_time = 0.0
        self._next_repeat_time = 0.0
        self._raw_word = ''
        self._suggestion = ''

    def update_swipe(self, nx: float, ny: float) -> None:
        """Call every frame while pinch is held, with normalised fingertip position.

        Parameters
        ----------
        nx, ny : float
            Normalised (0-1) x, y position of the index fingertip.
        """
        if not self._is_active:
            return

        self._path.append((nx, ny))
        key = self._nearest_key(nx, ny)
        if key is None:
            return

        now = time.monotonic()

        if key != self._last_key:
            # Moved to a new key
            self._append_key(key)
            self._last_key = key
            self._last_key_time = now
            self._next_repeat_time = now + self._hold_threshold
        else:
            # Same key – check hold-repeat
            if now >= self._next_repeat_time:
                self._append_key(key)
                self._next_repeat_time = now + self._hold_interval

        # Refresh suggestion after every update
        self._raw_word = ''.join(
            k.lower() for k in self._key_seq if len(k) == 1
        )
        self._suggestion = self._auto_correct(self._raw_word)

    def end_swipe(self) -> str:
        """Call when pinch is released.

        Returns
        -------
        str
            The auto-corrected word (or raw sequence if no correction found).
            Empty string if no keys were collected.
        """
        logger.debug('Swipe typing: END  raw=%r  suggestion=%r',
                     self._raw_word, self._suggestion)
        self._is_active = False
        word = self._suggestion if self._suggestion else self._raw_word
        self._completed_word = word
        return word

    def get_overlay_state(self) -> SwipeOverlayState:
        """Return a snapshot of current state for the UI overlay."""
        # Find which key is under the last fingertip position
        highlighted: Optional[str] = None
        if self._is_active and self._path:
            nx, ny = self._path[-1]
            highlighted = self._nearest_key(nx, ny)

        return SwipeOverlayState(
            is_active=self._is_active,
            key_cells=self._cells,
            highlighted_key=highlighted,
            path=list(self._path[-50:]),   # keep last 50 points for drawing
            raw_word=self._raw_word,
            suggestion=self._suggestion,
            completed_word=self._completed_word,
        )

    # ------------------------------------------------------------------
    # Keyboard geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cells() -> List[KeyCell]:
        """Construct the list of :class:`KeyCell` objects for the QWERTY layout."""
        cells: List[KeyCell] = []
        x_span = _KB_X_RIGHT - _KB_X_LEFT

        for row_idx, (row_keys, y_center) in enumerate(zip(QWERTY_ROWS, _ROW_Y)):
            n = len(row_keys)
            key_w = x_span / n
            x_half = key_w / 2.0

            for col_idx, label in enumerate(row_keys):
                x_center = _KB_X_LEFT + (col_idx + 0.5) * key_w
                cells.append(KeyCell(
                    label=label,
                    x_center=x_center,
                    y_center=y_center,
                    x_half=x_half,
                    y_half=_KEY_H_HALF,
                ))

        return cells

    def _nearest_key(self, nx: float, ny: float) -> Optional[str]:
        """Return the label of the key whose centre is closest to (nx, ny).

        Returns None when the finger is outside the keyboard zone entirely.
        """
        # Quick zone rejection: must be within extended keyboard bounding box
        y_top = _ROW_Y[0] - _KEY_H_HALF * 1.5
        y_bot = _ROW_Y[-1] + _KEY_H_HALF * 1.5
        if ny < y_top or ny > y_bot:
            return None

        best_key: Optional[str] = None
        best_dist = math.inf

        for cell in self._cells:
            # Normalise distances by cell half-extents (gives a "within cell" metric)
            dx = (nx - cell.x_center) / (cell.x_half * 1.5)
            dy = (ny - cell.y_center) / (cell.y_half * 1.5)
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_key = cell.label

        # Accept only if within 1.4× the normalised radius
        return best_key if best_dist <= 1.4 else None

    # ------------------------------------------------------------------
    # Key sequence helpers
    # ------------------------------------------------------------------

    def _append_key(self, key: str) -> None:
        """Add *key* to the sequence (handles SPACE / DEL / ENTER specially)."""
        if key == 'DEL':
            if self._key_seq:
                self._key_seq.pop()
        elif key in ('SPACE', 'ENTER'):
            # These terminate the swipe word; ignore mid-swipe
            pass
        else:
            self._key_seq.append(key)

    # ------------------------------------------------------------------
    # Auto-correction
    # ------------------------------------------------------------------

    def _auto_correct(self, raw: str) -> str:
        """Return the closest dictionary word for *raw* via Levenshtein distance.

        Returns the raw string unchanged if no good match is found.
        """
        if not raw or len(raw) < 2:
            return raw

        best_word = ''
        best_dist = math.inf

        for word in self._dictionary:
            # Quick length filter: skip words far longer/shorter than raw
            length_diff = abs(len(word) - len(raw))
            if length_diff > max(2, len(raw) // 2):
                continue

            dist = _levenshtein(raw, word)

            # Prefer word with lower distance; break ties by preferring closer length
            if dist < best_dist or (
                dist == best_dist
                and abs(len(word) - len(raw)) < abs(len(best_word) - len(raw))
            ):
                best_dist = dist
                best_word = word

        if not best_word:
            return raw

        max_allowed = max(1, int(len(raw) * self._max_dist_ratio))
        if best_dist <= max_allowed:
            return best_word

        return raw
