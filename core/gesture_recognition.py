"""Gesture Recognition – classify hand poses into named gestures.

Supported gestures
------------------
- ``point``      – index finger extended only  → normal mouse movement
- ``pinch``      – thumb + index tips close    → left click
- ``open_palm``  – all five fingers spread     → right click
- ``thumbs_up``  – thumb up, others curled     → execute last voice command
- ``peace``      – index + middle extended     → take screenshot
- ``fist``       – all fingers curled          → pause / freeze pointer
- ``swipe_right``– horizontal rightward motion → scroll right
- ``swipe_left`` – horizontal leftward motion  → scroll left
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
from collections import deque

from core.hand_tracking import HandData, HandLandmark


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GestureResult:
    name: str        # gesture identifier string
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _distance(a: HandLandmark, b: HandLandmark) -> float:
    """Euclidean distance in normalised-coordinate space."""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _finger_extended(tip: HandLandmark, pip: HandLandmark, mcp: HandLandmark) -> bool:
    """Return True when the finger is extended (tip above PIP joint in image coords)."""
    # In image coords y increases downward, so a raised fingertip has smaller y
    return tip.y < pip.y < mcp.y


# ---------------------------------------------------------------------------
# Recogniser
# ---------------------------------------------------------------------------

class GestureRecognizer:
    """Classify a :class:`~core.hand_tracking.HandData` snapshot into a gesture.

    Parameters
    ----------
    pinch_threshold : float
        Normalised distance below which thumb+index is "pinched" (default 0.06).
    cooldown : float
        Minimum seconds between two emissions of the same gesture.
    smoothing : int
        Number of frames to buffer for swipe detection.
    """

    # MediaPipe landmark indices
    _WRIST       = 0
    _THUMB_CMC   = 1
    _THUMB_MCP   = 2
    _THUMB_IP    = 3
    _THUMB_TIP   = 4
    _INDEX_MCP   = 5
    _INDEX_PIP   = 6
    _INDEX_DIP   = 7
    _INDEX_TIP   = 8
    _MIDDLE_MCP  = 9
    _MIDDLE_PIP  = 10
    _MIDDLE_DIP  = 11
    _MIDDLE_TIP  = 12
    _RING_MCP    = 13
    _RING_PIP    = 14
    _RING_DIP    = 15
    _RING_TIP    = 16
    _PINKY_MCP   = 17
    _PINKY_PIP   = 18
    _PINKY_DIP   = 19
    _PINKY_TIP   = 20

    def __init__(
        self,
        pinch_threshold: float = 0.06,
        cooldown: float = 0.5,
        smoothing: int = 6,
    ) -> None:
        self._pinch_threshold = pinch_threshold
        self._cooldown = cooldown
        self._last_gesture_time: dict[str, float] = {}

        # Circular buffer of (timestamp, index_tip_x) for swipe detection
        self._position_history: Deque[Tuple[float, float]] = deque(maxlen=smoothing)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, hand: HandData) -> GestureResult:
        """Return the most likely gesture for *hand*.

        Always returns a result; defaults to ``'none'`` when nothing matches.
        """
        lm = hand.landmarks
        if len(lm) < 21:
            return GestureResult(name='none', confidence=0.0)

        # ── Collect finger states ──────────────────────────────────────
        thumb_ext  = lm[self._THUMB_TIP].x < lm[self._THUMB_IP].x  # thumb points sideways
        index_ext  = _finger_extended(lm[self._INDEX_TIP],  lm[self._INDEX_PIP],  lm[self._INDEX_MCP])
        middle_ext = _finger_extended(lm[self._MIDDLE_TIP], lm[self._MIDDLE_PIP], lm[self._MIDDLE_MCP])
        ring_ext   = _finger_extended(lm[self._RING_TIP],   lm[self._RING_PIP],   lm[self._RING_MCP])
        pinky_ext  = _finger_extended(lm[self._PINKY_TIP],  lm[self._PINKY_PIP],  lm[self._PINKY_MCP])

        num_fingers = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        # ── Update position history ────────────────────────────────────
        now = time.time()
        idx_tip_x = lm[self._INDEX_TIP].x
        self._position_history.append((now, idx_tip_x))

        # ── Classify ──────────────────────────────────────────────────
        gesture = self._classify(
            lm=lm,
            thumb_ext=thumb_ext,
            index_ext=index_ext,
            middle_ext=middle_ext,
            ring_ext=ring_ext,
            pinky_ext=pinky_ext,
            num_fingers=num_fingers,
        )

        # ── Apply cooldown ────────────────────────────────────────────
        last = self._last_gesture_time.get(gesture, 0.0)
        if gesture not in ('point', 'none') and now - last < self._cooldown:
            # During cooldown, keep reporting 'point' or 'none' to avoid re-triggering
            gesture = 'point' if index_ext and num_fingers == 1 else 'none'

        if gesture not in ('point', 'none'):
            self._last_gesture_time[gesture] = now

        return GestureResult(name=gesture, timestamp=now)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        lm: List[HandLandmark],
        thumb_ext: bool,
        index_ext: bool,
        middle_ext: bool,
        ring_ext: bool,
        pinky_ext: bool,
        num_fingers: int,
    ) -> str:
        # ── Pinch (thumb + index close) ────────────────────────────────
        pinch_dist = _distance(lm[self._THUMB_TIP], lm[self._INDEX_TIP])
        if pinch_dist < self._pinch_threshold:
            return 'pinch'

        # ── Closed fist (all curled) ──────────────────────────────────
        if num_fingers == 0 and not thumb_ext:
            return 'fist'

        # ── Thumbs up (thumb up, all others curled) ───────────────────
        if (
            thumb_ext
            and not index_ext
            and not middle_ext
            and not ring_ext
            and not pinky_ext
            and lm[self._THUMB_TIP].y < lm[self._INDEX_MCP].y
        ):
            return 'thumbs_up'

        # ── Open palm (all five) ──────────────────────────────────────
        if num_fingers >= 4 and thumb_ext:
            return 'open_palm'

        # ── Peace sign (index + middle, others curled) ────────────────
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return 'peace'

        # ── Point (index only) ────────────────────────────────────────
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            # Check for horizontal swipe using position history
            swipe = self._detect_swipe()
            if swipe:
                return swipe
            return 'point'

        return 'none'

    def _detect_swipe(self) -> Optional[str]:
        """Return 'swipe_right', 'swipe_left', or None based on history."""
        if len(self._position_history) < 4:
            return None

        times = [t for t, _ in self._position_history]
        xs    = [x for _, x in self._position_history]

        elapsed = times[-1] - times[0]
        if elapsed < 0.05 or elapsed > 1.0:
            return None

        delta_x = xs[-1] - xs[0]
        # Threshold: at least 15% of frame width in under 1 s
        if abs(delta_x) < 0.15:
            return None

        # In a non-mirrored camera frame, hand moves right → x increases (delta_x > 0).
        # The mouse controller separately flips x for pointer movement; here we match
        # the natural screen-scroll convention: rightward hand motion → scroll right.
        return 'swipe_right' if delta_x > 0 else 'swipe_left'
