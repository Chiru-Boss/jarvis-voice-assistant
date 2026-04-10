"""Hand–Voice Integration – runs hand tracking in a background thread.

This module owns the hand-tracking loop and exposes a small API that
``main.py`` uses to:

1. Start / stop the background thread.
2. Query the latest gesture for combined hand + voice commands.
3. Handle gesture-triggered actions (click, screenshot, scroll …).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class HandVoiceIntegration:
    """Run hand tracking in a background thread alongside JARVIS voice.

    Parameters
    ----------
    camera_device : int
        Index of the webcam to use (0 = default).
    detection_confidence : float
        MediaPipe minimum detection confidence.
    tracking_confidence : float
        MediaPipe minimum tracking confidence.
    pinch_threshold : float
        Normalised thumb-index distance for a pinch click.
    smoothing : float
        EMA alpha for mouse smoothing.
    mouse_speed : float
        Pointer speed multiplier.
    click_delay : float
        Minimum seconds between clicks.
    gesture_cooldown : float
        Minimum seconds between non-movement gestures.
    show_overlay : bool
        Whether to display the OpenCV overlay window.
    on_voice_trigger : callable or None
        Callback invoked when the user makes a "thumbs up" gesture
        (signature: ``on_voice_trigger() -> None``).
    on_screenshot : callable or None
        Callback invoked on the "peace" gesture.
    """

    def __init__(
        self,
        camera_device: int = 0,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        pinch_threshold: float = 0.06,
        smoothing: float = 0.5,
        mouse_speed: float = 1.5,
        click_delay: float = 0.1,
        gesture_cooldown: float = 0.5,
        show_overlay: bool = True,
        on_voice_trigger: Optional[Callable] = None,
        on_screenshot:    Optional[Callable] = None,
        calibration_corners: Optional[list] = None,
    ) -> None:
        self._camera_device         = camera_device
        self._detection_confidence  = detection_confidence
        self._tracking_confidence   = tracking_confidence
        self._pinch_threshold       = pinch_threshold
        self._smoothing             = smoothing
        self._mouse_speed           = mouse_speed
        self._click_delay           = click_delay
        self._gesture_cooldown      = gesture_cooldown
        self._show_overlay          = show_overlay
        self._on_voice_trigger      = on_voice_trigger
        self._on_screenshot         = on_screenshot
        self._calibration_corners   = calibration_corners

        # Thread state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Latest gesture (shared with main thread)
        self._latest_gesture = 'none'
        self._gesture_lock   = threading.Lock()

        # Whether a hand is currently visible
        self._hand_visible = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background hand-tracking thread."""
        if self._thread and self._thread.is_alive():
            logger.warning('Hand tracking thread already running.')
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._tracking_loop, daemon=True, name='hand-tracking'
        )
        self._thread.start()
        logger.info('Hand tracking thread started.')

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info('Hand tracking thread stopped.')

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def hand_visible(self) -> bool:
        return self._hand_visible

    def get_latest_gesture(self) -> str:
        """Return the most recently detected gesture name."""
        with self._gesture_lock:
            return self._latest_gesture

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _tracking_loop(self) -> None:
        """Main loop: open camera, track hand, dispatch gesture actions."""
        # Late imports so that missing packages fail only when the thread
        # actually starts (not at import time).
        try:
            import cv2  # type: ignore
        except ImportError:
            logger.error(
                'opencv-python is not installed. '
                'Hand tracking disabled. Run: pip install opencv-python'
            )
            return

        try:
            from core.hand_tracking       import HandTracker
            from core.gesture_recognition  import GestureRecognizer
            from core.hand_mouse_controller import HandMouseController
            from core.hand_ui_overlay      import HandUIOverlay
        except Exception as exc:
            logger.error('Failed to import hand-tracking modules: %s', exc)
            return

        tracker    = HandTracker(
            max_num_hands=1,
            min_detection_confidence=self._detection_confidence,
            min_tracking_confidence=self._tracking_confidence,
        )
        recognizer = GestureRecognizer(
            pinch_threshold=self._pinch_threshold,
            cooldown=self._gesture_cooldown,
        )
        mouse_ctrl = HandMouseController(
            smoothing=self._smoothing,
            mouse_speed=self._mouse_speed,
            click_delay=self._click_delay,
        )
        if self._calibration_corners:
            mouse_ctrl.set_calibration(self._calibration_corners)
            logger.info('Calibration applied to mouse controller.')
        overlay    = HandUIOverlay() if self._show_overlay else None

        cap = cv2.VideoCapture(self._camera_device)
        if not cap.isOpened():
            logger.error('Could not open camera device %d.', self._camera_device)
            tracker.close()
            return

        logger.info('Hand tracking loop running (camera %d).', self._camera_device)

        gesture_name  = 'none'
        mouse_pos: Optional[tuple] = None

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning('Camera read failed; retrying…')
                    time.sleep(0.05)
                    continue

                hands = tracker.process_frame(frame)
                self._hand_visible = bool(hands)

                if hands:
                    hand = hands[0]

                    # ── Gesture recognition ──────────────────────────────
                    result = recognizer.recognize(hand)
                    gesture_name = result.name

                    with self._gesture_lock:
                        self._latest_gesture = gesture_name

                    # ── Mouse / gesture actions ──────────────────────────
                    self._dispatch_gesture(gesture_name, hand, mouse_ctrl)

                    # ── Update mouse position (pointer gesture) ──────────
                    if gesture_name in ('point', 'none') and not mouse_ctrl.is_paused:
                        idx_tip = hand.index_tip()
                        if idx_tip:
                            mouse_pos = mouse_ctrl.update_position(idx_tip.x, idx_tip.y)
                else:
                    gesture_name = 'none'
                    with self._gesture_lock:
                        self._latest_gesture = 'none'

                # ── Overlay rendering ──────────────────────────────────
                if overlay:
                    gesture_conf = hands[0].confidence if hands else 0.0
                    overlay.render(
                        frame=frame,
                        tracker=tracker,
                        hands=hands,
                        gesture_name=gesture_name,
                        gesture_confidence=gesture_conf,
                        mouse_pos=mouse_pos,
                        paused=mouse_ctrl.is_paused,
                    )
                    if overlay.should_quit():
                        logger.info('Overlay window closed by user.')
                        break

        finally:
            cap.release()
            tracker.close()
            if overlay:
                overlay.close()

    def _dispatch_gesture(self, gesture_name: str, hand, mouse_ctrl) -> None:
        """Execute the action associated with *gesture_name*."""
        idx_tip = hand.index_tip()

        if gesture_name == 'pinch':
            # Move to fingertip position first, then click
            if idx_tip:
                mouse_ctrl.update_position(idx_tip.x, idx_tip.y)
            mouse_ctrl.left_click()

        elif gesture_name == 'open_palm':
            if idx_tip:
                mouse_ctrl.update_position(idx_tip.x, idx_tip.y)
            mouse_ctrl.right_click()

        elif gesture_name == 'fist':
            if not mouse_ctrl.is_paused:
                mouse_ctrl.pause()
                logger.info('Pointer paused (fist gesture).')
        elif gesture_name == 'point' and mouse_ctrl.is_paused:
            mouse_ctrl.resume()
            logger.info('Pointer resumed (point gesture).')

        elif gesture_name == 'swipe_right':
            mouse_ctrl.scroll('right')

        elif gesture_name == 'swipe_left':
            mouse_ctrl.scroll('left')

        elif gesture_name == 'thumbs_up':
            logger.info('Thumbs-up detected – triggering voice command.')
            if self._on_voice_trigger:
                threading.Thread(
                    target=self._on_voice_trigger, daemon=True
                ).start()

        elif gesture_name == 'peace':
            logger.info('Peace sign detected – triggering screenshot.')
            if self._on_screenshot:
                threading.Thread(
                    target=self._on_screenshot, daemon=True
                ).start()
