"""UI Detector – locate UI elements on a screenshot using OCR.

Uses pytesseract to find text-based UI elements (search bars, buttons, input
fields) and returns their screen coordinates for mouse automation.

All heavy imports are lazy so the module loads even when pytesseract is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum pytesseract confidence (0-100) to consider a word real.
_CONFIDENCE_THRESHOLD = 30


class UIDetector:
    """Detect UI elements in a PIL screenshot image using OCR."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_search_bar(self, screenshot: Any) -> Optional[Tuple[int, int]]:
        """Return the (x, y) centre of a search or address bar, or ``None``.

        Searches for common search-bar keywords first; falls back to a
        heuristic position (top-centre of the screen) for Chromium browsers.
        """
        keywords = ['search', 'address', 'url', 'http', 'www', 'find']
        coords = self._find_by_keywords(screenshot, keywords)
        if coords:
            return coords

        # Heuristic fallback: Chromium address bars sit near the top-centre.
        try:
            width, height = screenshot.size
            return (width // 2, max(50, int(height * 0.04)))
        except Exception:
            return None

    def find_element_by_text(
        self, screenshot: Any, text: str
    ) -> Optional[Tuple[int, int]]:
        """Return (x, y) of a UI element whose OCR text contains *text*."""
        return self._find_by_keywords(screenshot, [text.lower()])

    def find_button(
        self, screenshot: Any, button_text: str
    ) -> Optional[Tuple[int, int]]:
        """Find a button with *button_text* and return its centre coordinates."""
        return self._find_by_keywords(screenshot, [button_text.lower()])

    def find_text_field(self, screenshot: Any) -> Optional[Tuple[int, int]]:
        """Find a generic input / text field on screen."""
        keywords = ['type here', 'enter text', 'search', 'input', 'query']
        return self._find_by_keywords(screenshot, keywords)

    def get_all_text_elements(
        self, screenshot: Any
    ) -> List[Dict[str, Any]]:
        """Return all detected text elements with their bounding boxes.

        Each element is a dict with keys: ``text``, ``x``, ``y``, ``w``,
        ``h``, ``confidence``.
        """
        elements: List[Dict[str, Any]] = []
        try:
            import pytesseract  # type: ignore

            data = pytesseract.image_to_data(
                screenshot, output_type=pytesseract.Output.DICT
            )
            for i, word in enumerate(data.get('text', [])):
                word = word.strip()
                if not word:
                    continue
                conf = int(data['conf'][i])
                if conf < _CONFIDENCE_THRESHOLD:
                    continue
                elements.append({
                    'text': word,
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'confidence': conf,
                })
        except ImportError:
            logger.debug('pytesseract not installed – UI detection unavailable.')
        except Exception as exc:
            logger.debug('get_all_text_elements failed: %s', exc)
        return elements

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_by_keywords(
        self, screenshot: Any, keywords: List[str]
    ) -> Optional[Tuple[int, int]]:
        """Return the centre of the first OCR word matching any keyword."""
        try:
            import pytesseract  # type: ignore

            data = pytesseract.image_to_data(
                screenshot, output_type=pytesseract.Output.DICT
            )
            for i, word in enumerate(data.get('text', [])):
                if not word.strip():
                    continue
                word_lower = word.lower()
                conf = int(data['conf'][i])
                if conf < _CONFIDENCE_THRESHOLD:
                    continue
                if any(kw in word_lower for kw in keywords):
                    cx = data['left'][i] + data['width'][i] // 2
                    cy = data['top'][i] + data['height'][i] // 2
                    return (cx, cy)
        except ImportError:
            logger.debug('pytesseract not installed – cannot locate UI elements.')
        except Exception as exc:
            logger.debug('_find_by_keywords failed: %s', exc)
        return None
