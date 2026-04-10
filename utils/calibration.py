"""Calibration helper – guides the user through marking screen corners.

Usage
-----
Run directly::

    python utils/calibration.py

or import and call :func:`run_calibration` from your code.  The calibration
data is persisted to a JSON file and can be loaded back with
:func:`load_calibration`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_CALIB_FILE = Path('jarvis_hand_calibration.json')

# Screen corners in display order
_CORNER_NAMES = [
    'TOP-LEFT corner',
    'TOP-RIGHT corner',
    'BOTTOM-LEFT corner',
    'BOTTOM-RIGHT corner',
]


def run_calibration(
    camera_device: int = 0,
    detection_confidence: float = 0.7,
    tracking_confidence: float = 0.5,
    save_path: Path = _CALIB_FILE,
) -> Optional[List[Tuple[float, float]]]:
    """Run the interactive calibration wizard.

    The user holds their index finger at each screen corner when prompted.
    Captured positions are averaged over a short window to reduce jitter.

    Returns
    -------
    list of (x, y) or None
        Four normalised corner positions [TL, TR, BL, BR], or None on failure.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.error('opencv-python is required for calibration.')
        return None

    try:
        from core.hand_tracking import HandTracker
    except ImportError as exc:
        logger.error('Failed to import HandTracker: %s', exc)
        return None

    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )

    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        logger.error('Could not open camera %d for calibration.', camera_device)
        tracker.close()
        return None

    corners: List[Tuple[float, float]] = []

    try:
        for corner_name in _CORNER_NAMES:
            print(f'\n📍 Point your index finger at the {corner_name}.')
            print('   Hold still for 2 seconds… ', end='', flush=True)

            samples: List[Tuple[float, float]] = []
            deadline = time.time() + 4.0   # 4 s window (first 2 s warm-up, 2 s capture)

            while time.time() < deadline:
                ret, frame = cap.read()
                if not ret:
                    continue

                hands = tracker.process_frame(frame)
                if not hands:
                    continue

                idx_tip = hands[0].index_tip()
                if idx_tip is None:
                    continue

                elapsed = deadline - time.time()
                if elapsed < 2.0:          # last 2 s → collect samples
                    samples.append((idx_tip.x, idx_tip.y))

                # Simple overlay
                h, w = frame.shape[:2]
                cv2.putText(
                    frame,
                    f'Point at {corner_name}',
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow('JARVIS Calibration', frame)
                cv2.waitKey(1)

            if not samples:
                print('❌ No hand detected. Calibration aborted.')
                return None

            avg_x = sum(s[0] for s in samples) / len(samples)
            avg_y = sum(s[1] for s in samples) / len(samples)
            corners.append((avg_x, avg_y))
            print(f'✅ Captured ({avg_x:.3f}, {avg_y:.3f})')

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    _save_calibration(corners, save_path)
    print(f'\n✅ Calibration saved to {save_path}')
    return corners


def load_calibration(
    save_path: Path = _CALIB_FILE,
) -> Optional[List[Tuple[float, float]]]:
    """Load previously saved calibration data.

    Returns
    -------
    list of (x, y) or None
        Four corner positions, or None if the file does not exist or is invalid.
    """
    if not save_path.exists():
        return None
    try:
        data = json.loads(save_path.read_text())
        corners = [tuple(pt) for pt in data['corners']]
        if len(corners) != 4:
            return None
        return corners  # type: ignore[return-value]
    except Exception as exc:
        logger.warning('Failed to load calibration: %s', exc)
        return None


def _save_calibration(
    corners: List[Tuple[float, float]],
    save_path: Path,
) -> None:
    """Persist *corners* to *save_path* as JSON."""
    data = {'corners': [list(c) for c in corners]}
    save_path.write_text(json.dumps(data, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    result = run_calibration()
    if result:
        print('\nCalibration corners:')
        for name, corner in zip(_CORNER_NAMES, result):
            print(f'  {name}: {corner}')
