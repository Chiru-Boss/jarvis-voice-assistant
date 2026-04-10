"""Hand UI Overlay – real-time webcam feed with hand skeleton and HUD.

Draws the MediaPipe hand skeleton, gesture label, confidence, FPS counter,
and mouse-pointer position over the live webcam frame in an OpenCV window.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from core.hand_tracking import HandData, HandTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_GREEN   = (0,   255,  0)
_CYAN    = (255, 255,  0)
_WHITE   = (255, 255, 255)
_BLACK   = (0,   0,    0)
_RED     = (0,   0,    255)
_YELLOW  = (0,   255,  255)
_MAGENTA = (255, 0,    255)


class HandUIOverlay:
    """Manage the overlay window that shows the hand-tracking feed.

    Parameters
    ----------
    window_name : str
        Title of the OpenCV window.
    show_fps : bool
        Whether to display the frames-per-second counter.
    """

    def __init__(
        self,
        window_name: str = '🤖 JARVIS – Hand Control',
        show_fps: bool = True,
    ) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                'opencv-python is required for the overlay. '
                'Install it with: pip install opencv-python'
            ) from exc

        self._cv2 = cv2
        self._window_name = window_name
        self._show_fps    = show_fps

        # FPS calculation
        self._fps_t0  = time.time()
        self._fps_cnt = 0
        self._fps     = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        frame,
        tracker: HandTracker,
        hands: list,
        gesture_name: str = '',
        gesture_confidence: float = 1.0,
        mouse_pos: Optional[Tuple[int, int]] = None,
        paused: bool = False,
    ):
        """Draw overlay elements and display the frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR webcam frame (modified in-place).
        tracker : HandTracker
            Used to draw skeleton landmarks.
        hands : list[HandData]
            Detected hands for the current frame.
        gesture_name : str
            Currently detected gesture label.
        gesture_confidence : float
            Confidence of the detected gesture.
        mouse_pos : (int, int) or None
            Current mouse pixel position to display.
        paused : bool
            When True, shows a PAUSED banner.
        """
        cv2 = self._cv2

        # ── Draw skeleton for each hand ────────────────────────────────
        for hand in hands:
            tracker.draw_landmarks(frame, hand)
            self._draw_fingertip_markers(frame, hand)

        # ── HUD elements ───────────────────────────────────────────────
        self._draw_hud(
            frame,
            gesture_name=gesture_name,
            gesture_confidence=gesture_confidence,
            mouse_pos=mouse_pos,
            paused=paused,
        )

        # ── FPS ────────────────────────────────────────────────────────
        if self._show_fps:
            self._update_fps()
            self._draw_fps(frame)

        # ── Show ───────────────────────────────────────────────────────
        cv2.imshow(self._window_name, frame)

    def should_quit(self, wait_ms: int = 1) -> bool:
        """Return True if the user pressed 'q' or closed the window."""
        cv2 = self._cv2
        key = cv2.waitKey(wait_ms) & 0xFF
        return key == ord('q')

    def close(self) -> None:
        """Destroy the overlay window."""
        try:
            self._cv2.destroyWindow(self._window_name)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_fingertip_markers(self, frame, hand: HandData) -> None:
        """Draw bright circles on all five fingertips."""
        cv2 = self._cv2
        h, w = frame.shape[:2]
        tips = [hand.thumb_tip(), hand.index_tip(), hand.middle_tip(),
                hand.ring_tip(), hand.pinky_tip()]
        for tip in tips:
            if tip is None:
                continue
            px = int(tip.x * w)
            py = int(tip.y * h)
            cv2.circle(frame, (px, py), 8, _CYAN, -1)
            cv2.circle(frame, (px, py), 10, _WHITE, 2)

    def _draw_hud(
        self,
        frame,
        gesture_name: str,
        gesture_confidence: float,
        mouse_pos: Optional[Tuple[int, int]],
        paused: bool,
    ) -> None:
        cv2 = self._cv2
        h, w = frame.shape[:2]

        # ── Semi-transparent HUD background ───────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), _BLACK, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # ── Gesture label ──────────────────────────────────────────────
        gesture_display = gesture_name.replace('_', ' ').upper() if gesture_name else 'NONE'
        cv2.putText(
            frame,
            f'Gesture: {gesture_display}',
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _GREEN,
            2,
        )

        # ── Confidence bar ─────────────────────────────────────────────
        bar_w = int(gesture_confidence * 150)
        cv2.rectangle(frame, (10, 38), (160, 52), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 38), (10 + bar_w, 52), _GREEN, -1)
        cv2.putText(
            frame,
            f'{gesture_confidence * 100:.0f}%',
            (165, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _WHITE,
            1,
        )

        # ── Mouse position ─────────────────────────────────────────────
        if mouse_pos:
            cv2.putText(
                frame,
                f'Mouse: ({mouse_pos[0]}, {mouse_pos[1]})',
                (w - 220, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                _YELLOW,
                1,
            )

        # ── PAUSED banner ──────────────────────────────────────────────
        if paused:
            txt = 'PAUSED'
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
            cx = (w - tw) // 2
            cy = (h + th) // 2
            cv2.putText(frame, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.0, _RED, 4)

        # ── Gesture guide (bottom strip) ──────────────────────────────
        guide_y = h - 10
        guide = (
            'Pinch=Click  Palm=RClick  Peace=Screenshot  '
            'ThumbUp=Cmd  Fist=Pause  Swipe=Scroll'
        )
        cv2.putText(
            frame,
            guide,
            (5, guide_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            _WHITE,
            1,
        )

    def _update_fps(self) -> None:
        self._fps_cnt += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 1.0:
            self._fps = self._fps_cnt / elapsed
            self._fps_cnt = 0
            self._fps_t0  = time.time()

    def _draw_fps(self, frame) -> None:
        cv2 = self._cv2
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f'FPS: {self._fps:.1f}',
            (w - 100, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _MAGENTA,
            1,
        )
