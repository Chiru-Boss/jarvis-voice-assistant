"""Hand Tracking configuration – environment-driven settings."""

from __future__ import annotations

import os


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


HAND_TRACKING_CONFIG = {
    # ── Feature toggle ──────────────────────────────────────────────────
    'HAND_TRACKING_ENABLED': os.getenv('HAND_TRACKING_ENABLED', 'false').lower() == 'true',

    # ── Camera ──────────────────────────────────────────────────────────
    # Index of the webcam device to use (0 = default system webcam).
    'CAMERA_DEVICE': _safe_int(os.getenv('CAMERA_DEVICE'), default=0),

    # ── MediaPipe detection parameters ──────────────────────────────────
    # Minimum confidence for hand detection to be considered successful.
    'HAND_DETECTION_CONFIDENCE': _safe_float(
        os.getenv('HAND_DETECTION_CONFIDENCE'), default=0.7
    ),
    # Minimum confidence for landmark tracking to be considered successful.
    'HAND_TRACKING_CONFIDENCE': _safe_float(
        os.getenv('HAND_TRACKING_CONFIDENCE'), default=0.5
    ),
    # Maximum number of hands to detect simultaneously.
    'MAX_NUM_HANDS': _safe_int(os.getenv('MAX_NUM_HANDS'), default=1),

    # ── Gesture smoothing ────────────────────────────────────────────────
    # EMA alpha for position smoothing (0 = no movement, 1 = raw position).
    'GESTURE_SMOOTHING': _safe_float(os.getenv('GESTURE_SMOOTHING'), default=0.5),

    # ── Click behaviour ──────────────────────────────────────────────────
    # Seconds to wait between consecutive click events.
    'CLICK_DELAY': _safe_float(os.getenv('CLICK_DELAY'), default=0.1),

    # ── Calibration ──────────────────────────────────────────────────────
    # When True, launch calibration wizard on startup.
    'CALIBRATION_MODE': os.getenv('CALIBRATION_MODE', 'false').lower() == 'true',

    # ── Overlay ──────────────────────────────────────────────────────────
    # Display the webcam + hand-skeleton overlay window.
    'SHOW_HAND_OVERLAY': os.getenv('SHOW_HAND_OVERLAY', 'true').lower() == 'true',

    # ── Mouse control ────────────────────────────────────────────────────
    # Multiplier applied to the raw finger movement (higher = faster pointer).
    'MOUSE_SPEED': _safe_float(os.getenv('MOUSE_SPEED'), default=1.5),

    # ── Gesture cooldown ─────────────────────────────────────────────────
    # Minimum seconds between two gesture recognitions of the same type.
    'GESTURE_COOLDOWN': _safe_float(os.getenv('GESTURE_COOLDOWN'), default=0.5),

    # ── Pinch threshold ──────────────────────────────────────────────────
    # Normalised distance (0-1) below which thumb+index is considered "pinched".
    'PINCH_THRESHOLD': _safe_float(os.getenv('PINCH_THRESHOLD'), default=0.06),
}
