"""Hand Tracking Module – real-time hand detection using MediaPipe.

Detects up to *MAX_NUM_HANDS* hands from a webcam feed and returns 21
normalised landmark positions per hand together with helper geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HandLandmark:
    """Single 3-D landmark (x, y are normalised 0-1, z is relative depth)."""
    x: float
    y: float
    z: float


@dataclass
class HandData:
    """All information extracted from a single detected hand."""

    # 21 landmarks (MediaPipe ordering)
    landmarks: List[HandLandmark] = field(default_factory=list)

    # 'Left' or 'Right' as reported by MediaPipe (mirrored for selfie-view)
    handedness: str = 'Unknown'

    # Overall detection confidence [0, 1]
    confidence: float = 0.0

    # Bounding box in normalised coords (x_min, y_min, x_max, y_max)
    bounding_box: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    # Whether this hand is currently visible
    visible: bool = True

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    def index_tip(self) -> Optional[HandLandmark]:
        """Return the index-finger tip landmark (index 8)."""
        return self.landmarks[8] if len(self.landmarks) > 8 else None

    def thumb_tip(self) -> Optional[HandLandmark]:
        """Return the thumb tip landmark (index 4)."""
        return self.landmarks[4] if len(self.landmarks) > 4 else None

    def middle_tip(self) -> Optional[HandLandmark]:
        """Return the middle-finger tip landmark (index 12)."""
        return self.landmarks[12] if len(self.landmarks) > 12 else None

    def ring_tip(self) -> Optional[HandLandmark]:
        """Return the ring-finger tip landmark (index 16)."""
        return self.landmarks[16] if len(self.landmarks) > 16 else None

    def pinky_tip(self) -> Optional[HandLandmark]:
        """Return the pinky tip landmark (index 20)."""
        return self.landmarks[20] if len(self.landmarks) > 20 else None

    def wrist(self) -> Optional[HandLandmark]:
        """Return the wrist landmark (index 0)."""
        return self.landmarks[0] if self.landmarks else None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class HandTracker:
    """Wraps MediaPipe Hands for frame-by-frame hand detection.

    Parameters
    ----------
    max_num_hands : int
        Maximum number of hands to detect per frame.
    min_detection_confidence : float
        Minimum confidence to accept a new detection.
    min_tracking_confidence : float
        Minimum confidence to continue tracking an existing hand.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                'mediapipe is required for hand tracking. '
                'Install it with: pip install mediapipe'
            ) from exc

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr) -> List[HandData]:
        """Detect hands in *frame_bgr* (OpenCV BGR image).

        Returns
        -------
        list[HandData]
            One entry per detected hand; empty when no hands found.
        """
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                'opencv-python is required for hand tracking. '
                'Install it with: pip install opencv-python'
            ) from exc

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return []

        hands: List[HandData] = []
        handedness_list = results.multi_handedness or []

        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            landmarks = [
                HandLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_lm.landmark
            ]

            # Bounding box
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            bb = (min(xs), min(ys), max(xs), max(ys))

            # Handedness
            side = 'Unknown'
            confidence = 0.0
            if idx < len(handedness_list):
                clf = handedness_list[idx].classification[0]
                side = clf.label
                confidence = clf.score

            hands.append(
                HandData(
                    landmarks=landmarks,
                    handedness=side,
                    confidence=confidence,
                    bounding_box=bb,
                )
            )

        return hands

    def draw_landmarks(self, frame_bgr, hand_data: HandData):
        """Draw the hand skeleton overlay onto *frame_bgr* in-place.

        Returns the annotated frame.
        """
        try:
            import mediapipe as mp  # type: ignore
        except ImportError:
            return frame_bgr

        # Reconstruct a MediaPipe NormalizedLandmarkList for drawing
        from mediapipe.framework.formats import landmark_pb2  # type: ignore

        hand_lm_list = landmark_pb2.NormalizedLandmarkList()
        for lm in hand_data.landmarks:
            landmark = hand_lm_list.landmark.add()
            landmark.x = lm.x
            landmark.y = lm.y
            landmark.z = lm.z

        self._mp_drawing.draw_landmarks(
            frame_bgr,
            hand_lm_list,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_drawing_styles.get_default_hand_landmarks_style(),
            self._mp_drawing_styles.get_default_hand_connections_style(),
        )
        return frame_bgr

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
