"""Hand Mouse Controller – map normalised finger position to screen coords.

The index-finger tip position (normalised 0-1 within the camera frame) is
converted to absolute screen pixel coordinates using an optional calibration
mapping.  An EMA smoothing filter reduces jitter before the pointer is moved.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _get_screen_size() -> Tuple[int, int]:
    """Return (width, height) of the primary screen."""
    try:
        import pyautogui  # type: ignore
        return pyautogui.size()
    except Exception:
        return (1920, 1080)


class HandMouseController:
    """Translate normalised hand-landmark coordinates into mouse actions.

    Parameters
    ----------
    smoothing : float
        EMA alpha [0, 1].  Lower = smoother but laggier; higher = more
        responsive but jittery.
    mouse_speed : float
        Scaling factor applied to the raw movement (> 1 = faster).
    click_delay : float
        Minimum seconds between consecutive click events.
    screen_width / screen_height : int
        Override the auto-detected screen resolution.
    """

    def __init__(
        self,
        smoothing: float = 0.5,
        mouse_speed: float = 1.5,
        click_delay: float = 0.1,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
    ) -> None:
        sw, sh = _get_screen_size()
        self._screen_w = screen_width  or sw
        self._screen_h = screen_height or sh

        self._smoothing   = max(0.01, min(1.0, smoothing))
        self._mouse_speed = mouse_speed
        self._click_delay = click_delay

        # Smoothed pointer position (0-1 normalised)
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None

        # Timestamps for click debouncing
        self._last_left_click  = 0.0
        self._last_right_click = 0.0
        self._last_scroll      = 0.0

        # Calibration corners (normalised): top-left, top-right, bottom-left, bottom-right
        self._cal_corners: Optional[list] = None

        # Pointer paused (fist gesture)
        self._paused = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def set_calibration(self, corners: list) -> None:
        """Store four calibration corner points.

        Parameters
        ----------
        corners : list of (x, y) tuples
            Normalised coordinates [0, 1] for TL, TR, BL, BR.
        """
        if len(corners) == 4:
            self._cal_corners = list(corners)
            logger.info('Calibration set: %s', corners)

    def clear_calibration(self) -> None:
        """Remove any previously set calibration."""
        self._cal_corners = None

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def update_position(self, norm_x: float, norm_y: float) -> Optional[Tuple[int, int]]:
        """Smooth *norm_x/norm_y* and move the system mouse pointer.

        Parameters
        ----------
        norm_x, norm_y : float
            Normalised finger position in [0, 1].

        Returns
        -------
        (pixel_x, pixel_y) after mapping, or None if paused.
        """
        if self._paused:
            return None

        # Apply calibration mapping if available
        if self._cal_corners:
            norm_x, norm_y = self._apply_calibration(norm_x, norm_y)

        # EMA smoothing
        if self._smooth_x is None:
            self._smooth_x = norm_x
            self._smooth_y = norm_y
        else:
            a = self._smoothing
            self._smooth_x = a * norm_x + (1 - a) * self._smooth_x
            self._smooth_y = a * norm_y + (1 - a) * self._smooth_y

        # Clamp to [0, 1]
        sx = max(0.0, min(1.0, self._smooth_x))
        sy = max(0.0, min(1.0, self._smooth_y))

        # Flip x for mirror (selfie-camera convention)
        sx = 1.0 - sx

        # Map to screen pixels
        px = int(sx * self._screen_w)
        py = int(sy * self._screen_h)

        try:
            import pyautogui  # type: ignore
            pyautogui.moveTo(px, py, duration=0)
        except Exception as exc:
            logger.debug('Mouse move failed: %s', exc)

        return (px, py)

    def pause(self) -> None:
        """Freeze the pointer (fist gesture)."""
        self._paused = True

    def resume(self) -> None:
        """Resume pointer movement."""
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------------------
    # Click / scroll
    # ------------------------------------------------------------------

    def left_click(self) -> bool:
        """Perform a left click if enough time has elapsed since the last one."""
        now = time.time()
        if now - self._last_left_click < self._click_delay:
            return False
        self._last_left_click = now
        try:
            import pyautogui  # type: ignore
            pyautogui.click()
            logger.debug('Left click performed')
            return True
        except Exception as exc:
            logger.debug('Left click failed: %s', exc)
            return False

    def right_click(self) -> bool:
        """Perform a right click if enough time has elapsed since the last one."""
        now = time.time()
        if now - self._last_right_click < self._click_delay:
            return False
        self._last_right_click = now
        try:
            import pyautogui  # type: ignore
            pyautogui.rightClick()
            logger.debug('Right click performed')
            return True
        except Exception as exc:
            logger.debug('Right click failed: %s', exc)
            return False

    def scroll(self, direction: str, amount: int = 3) -> bool:
        """Scroll in *direction* ('right' or 'left') by *amount* clicks."""
        now = time.time()
        if now - self._last_scroll < self._click_delay:
            return False
        self._last_scroll = now
        try:
            import pyautogui  # type: ignore
            if direction == 'right':
                pyautogui.hscroll(amount)
            else:
                pyautogui.hscroll(-amount)
            logger.debug('Scroll %s performed', direction)
            return True
        except Exception as exc:
            logger.debug('Scroll failed: %s', exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_calibration(self, x: float, y: float) -> Tuple[float, float]:
        """Bi-linear interpolation using the four calibration corners."""
        (tlx, tly), (trx, tr_y), (blx, bly), (brx, bry) = self._cal_corners  # type: ignore[misc]

        # Width / height at the y-level
        top_x   = tlx + (trx - tlx) * x
        bot_x   = blx + (brx - blx) * x
        mapped_x = top_x + (bot_x - top_x) * y

        left_y  = tly + (bly - tly) * y
        right_y = tr_y + (bry - tr_y) * y
        mapped_y = left_y + (right_y - left_y) * x

        # Normalise to [0, 1]
        all_x = [tlx, trx, blx, brx]
        all_y = [tly, tr_y, bly, bry]
        span_x = max(all_x) - min(all_x)
        span_y = max(all_y) - min(all_y)
        if span_x > 0:
            mapped_x = (mapped_x - min(all_x)) / span_x
        if span_y > 0:
            mapped_y = (mapped_y - min(all_y)) / span_y

        return (mapped_x, mapped_y)
