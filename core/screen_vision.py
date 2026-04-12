"""Screen Vision – screenshot capture, OCR, UI element detection, and change detection.

Provides on-demand and cached screenshot capabilities so JARVIS can "see"
what is currently on screen.  OCR is performed with pytesseract when available,
falling back to a graceful degradation message.  OpenCV is used only when
available (already a project dependency for hand tracking).

Usage
-----
    from core.screen_vision import ScreenVision
    vision = ScreenVision()
    info = vision.get_screen_content()
    print(info['ocr_text'])
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum seconds between automatic (cached) captures to avoid excessive CPU.
_DEFAULT_CACHE_TTL = 2.0
# Maximum text length returned from OCR before truncation.
_MAX_OCR_CHARS = 3000


class ScreenVision:
    """Capture the screen and extract text / UI element information.

    Parameters
    ----------
    cache_ttl : float
        Seconds before a cached screenshot is considered stale (default 2 s).
    """

    def __init__(self, cache_ttl: float = _DEFAULT_CACHE_TTL) -> None:
        self._cache_ttl = cache_ttl
        self._last_capture_time: float = 0.0
        self._last_image_bytes: Optional[bytes] = None
        self._last_image_hash: Optional[str] = None
        self._last_ocr: str = ''
        self._change_detected: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self, force: bool = False) -> Optional[Any]:
        """Return a PIL Image of the current screen (cached unless *force*).

        Returns None if Pillow is not installed.
        """
        now = time.monotonic()
        if not force and (now - self._last_capture_time) < self._cache_ttl:
            # Reconstruct from cached bytes if possible.
            if self._last_image_bytes:
                try:
                    from PIL import Image  # type: ignore
                    return Image.open(io.BytesIO(self._last_image_bytes))
                except Exception:
                    pass

        try:
            from PIL import ImageGrab  # type: ignore
        except ImportError:
            logger.warning('Pillow not installed – screen capture unavailable.')
            return None

        try:
            img = ImageGrab.grab()
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            raw = buf.getvalue()

            new_hash = hashlib.md5(raw).hexdigest()
            self._change_detected = new_hash != self._last_image_hash

            self._last_image_bytes = raw
            self._last_image_hash = new_hash
            self._last_capture_time = time.monotonic()
            return img
        except Exception as exc:
            logger.error('Screen capture failed: %s', exc)
            return None

    def get_ocr_text(self, image: Any = None, force: bool = False) -> str:
        """Extract text from *image* (or a fresh capture) using pytesseract.

        Returns an empty string if pytesseract is not installed.
        """
        if image is None:
            image = self.capture(force=force)
        if image is None:
            return ''

        # Try pytesseract first, then fall back to easyocr.
        try:
            import pytesseract  # type: ignore
            text = pytesseract.image_to_string(image)
            self._last_ocr = text.strip()
            return self._last_ocr
        except ImportError:
            pass
        except Exception as exc:
            logger.debug('pytesseract error: %s', exc)

        try:
            import easyocr  # type: ignore
            import numpy as np  # type: ignore
            reader = easyocr.Reader(['en'], verbose=False)
            results = reader.readtext(np.array(image))
            text = ' '.join(r[1] for r in results)
            self._last_ocr = text.strip()
            return self._last_ocr
        except ImportError:
            pass
        except Exception as exc:
            logger.debug('easyocr error: %s', exc)

        return ''

    def get_screen_content(self, force: bool = False) -> Dict[str, Any]:
        """Return a dict with screenshot path, OCR text, dimensions, and change flag.

        This is the primary method called by the MCP tool.
        """
        img = self.capture(force=force)
        result: Dict[str, Any] = {
            'captured': img is not None,
            'screen_changed': self._change_detected,
            'ocr_text': '',
            'width': 0,
            'height': 0,
            'elements': [],
        }
        if img is None:
            result['error'] = 'Screen capture unavailable (Pillow not installed).'
            return result

        result['width'], result['height'] = img.size

        ocr = self.get_ocr_text(image=img)
        result['ocr_text'] = ocr[:_MAX_OCR_CHARS] if len(ocr) > _MAX_OCR_CHARS else ocr

        result['elements'] = self._find_ui_elements(img)
        return result

    def has_screen_changed(self) -> bool:
        """Return True if the screen changed since the last capture."""
        return self._change_detected

    def save_screenshot(self, path: str = 'jarvis_screenshot.png') -> str:
        """Capture the screen and save to *path*. Returns a status string."""
        img = self.capture(force=True)
        if img is None:
            return '❌ Screen capture unavailable.'
        try:
            img.save(path)
            return f"✅ Screenshot saved to '{path}'."
        except Exception as exc:
            return f"❌ Could not save screenshot: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_ui_elements(self, image: Any) -> List[Dict[str, Any]]:
        """Detect simple rectangular UI elements using OpenCV contour detection.

        Returns a list of dicts with ``x``, ``y``, ``w``, ``h`` keys.
        Falls back to an empty list if OpenCV is not available.
        """
        elements: List[Dict[str, Any]] = []
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            img_array = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            # Edge detection to find UI boundaries.
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Ignore very small or very large contours.
                if area < 500 or area > (image.width * image.height * 0.5):
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                elements.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

            # Limit output to the 20 largest elements.
            elements.sort(key=lambda e: e['w'] * e['h'], reverse=True)
            elements = elements[:20]
        except ImportError:
            pass
        except Exception as exc:
            logger.debug('UI element detection failed: %s', exc)

        return elements
