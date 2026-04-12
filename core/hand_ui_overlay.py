"""Hand UI Overlay – real-time webcam feed with hand skeleton and HUD.

Draws the MediaPipe hand skeleton, gesture label, confidence, FPS counter,
mouse-pointer position, and the swipe-keyboard overlay over the live webcam
frame in an OpenCV window.
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
_ORANGE  = (0,   165,  255)
_BLUE    = (255, 100,  30)


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
        swipe_state=None,
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
        swipe_state : SwipeOverlayState or None
            Current swipe-keyboard state; when provided the keyboard overlay is drawn.
        """
        cv2 = self._cv2

        # ── Swipe keyboard overlay (drawn before skeleton so keys appear underneath)
        if swipe_state is not None:
            self._draw_keyboard(frame, swipe_state)

        # ── Draw skeleton for each hand ────────────────────────────────
        for hand in hands:
            tracker.draw_landmarks(frame, hand)
            self._draw_fingertip_markers(frame, hand)

        # ── Swipe path & word (drawn after skeleton so it appears on top)
        if swipe_state is not None and swipe_state.is_active:
            self._draw_swipe_path(frame, swipe_state)
            self._draw_swipe_word(frame, swipe_state)

        # ── HUD elements ───────────────────────────────────────────────
        self._draw_hud(
            frame,
            gesture_name=gesture_name,
            gesture_confidence=gesture_confidence,
            mouse_pos=mouse_pos,
            paused=paused,
            swipe_active=(swipe_state is not None and swipe_state.is_active),
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
        swipe_active: bool = False,
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
        if swipe_active:
            guide = 'SWIPE TYPING  –  Hold key >1s to repeat  |  Release pinch to confirm word'
            guide_color = _ORANGE
        else:
            guide = (
                'Pinch=Swipe-Type  Palm=RClick  Peace=Screenshot  '
                'ThumbUp=Cmd  Fist=Pause  Swipe=Scroll'
            )
            guide_color = _WHITE
        cv2.putText(
            frame,
            guide,
            (5, guide_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            guide_color,
            1,
        )

    def _draw_keyboard(self, frame, swipe_state) -> None:
        """Draw the QWERTY keyboard overlay at the bottom of *frame*."""
        cv2 = self._cv2
        h, w = frame.shape[:2]

        # ── Semi-transparent keyboard background ───────────────────────
        kb_overlay = frame.copy()
        # Keyboard starts at the first row's top y
        top_y_norm = swipe_state.key_cells[0].y_center - swipe_state.key_cells[0].y_half * 2
        top_y_px = int(top_y_norm * h)
        cv2.rectangle(kb_overlay, (0, top_y_px), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(kb_overlay, 0.55, frame, 0.45, 0, frame)

        # ── Draw each key ──────────────────────────────────────────────
        for cell in swipe_state.key_cells:
            cx = int(cell.x_center * w)
            cy = int(cell.y_center * h)
            half_w = int(cell.x_half * w) - 2   # small gap between keys
            half_h = int(cell.y_half * h)

            is_highlighted = (
                swipe_state.is_active
                and cell.label == swipe_state.highlighted_key
            )

            if is_highlighted:
                bg_color  = _ORANGE
                txt_color = _BLACK
                thickness = -1
            else:
                bg_color  = (60, 60, 60)
                txt_color = _WHITE
                thickness = -1

            # Key background
            pt1 = (cx - half_w, cy - half_h)
            pt2 = (cx + half_w, cy + half_h)
            cv2.rectangle(frame, pt1, pt2, bg_color, thickness)
            cv2.rectangle(frame, pt1, pt2, (120, 120, 120), 1)

            # Key label (abbreviated for wide keys)
            label = cell.label
            if label == 'SPACE':
                label = 'SPC'
            elif label == 'ENTER':
                label = 'ENT'
            font_scale = 0.38 if len(label) > 1 else 0.45
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = cx - tw // 2
            ty = cy + th // 2
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, 1)

    def _draw_swipe_path(self, frame, swipe_state) -> None:
        """Draw the finger's swipe trail on *frame*."""
        cv2 = self._cv2
        h, w = frame.shape[:2]
        path = swipe_state.path
        if len(path) < 2:
            return

        for i in range(1, len(path)):
            x0, y0 = int(path[i - 1][0] * w), int(path[i - 1][1] * h)
            x1, y1 = int(path[i][0] * w),     int(path[i][1] * h)
            # Gradient from cyan (older) to orange (newer)
            t = i / len(path)
            b = int(255 * (1 - t))
            g = int(200 * (1 - t) + 165 * t)
            r = int(255 * t)
            color = (b, g, r)
            cv2.line(frame, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)

        # Draw a bright dot at the current fingertip position
        lx, ly = int(path[-1][0] * w), int(path[-1][1] * h)
        cv2.circle(frame, (lx, ly), 7, _ORANGE, -1)
        cv2.circle(frame, (lx, ly), 9, _WHITE, 2)

    def _draw_swipe_word(self, frame, swipe_state) -> None:
        """Draw the current raw word and correction suggestion on *frame*."""
        cv2 = self._cv2
        h, w = frame.shape[:2]

        raw        = swipe_state.raw_word.upper()
        suggestion = swipe_state.suggestion.upper()

        # Box in the middle of the frame, above the keyboard
        box_y = int(0.55 * h)
        box_x = int(0.02 * w)

        bg = frame.copy()
        cv2.rectangle(bg, (box_x, box_y - 30), (box_x + int(w * 0.96), box_y + 10), _BLACK, -1)
        cv2.addWeighted(bg, 0.5, frame, 0.5, 0, frame)

        # Raw sequence
        cv2.putText(
            frame,
            f'Typing: {raw}',
            (box_x + 4, box_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _CYAN,
            1,
        )

        # Suggestion (only if different from raw)
        if suggestion and suggestion != raw:
            cv2.putText(
                frame,
                f'  → {suggestion}',
                (box_x + 4 + int(w * 0.38), box_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _GREEN,
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
