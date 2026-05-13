from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TrayController:
    """System tray status integration with graceful fallback."""

    def __init__(self, tooltip: str = 'JARVIS v3') -> None:
        self.tooltip = tooltip
        self.status = 'Idle'
        self._icon = None

    def start(self) -> bool:
        try:
            import pystray  # type: ignore
            from PIL import Image, ImageDraw  # type: ignore
        except ImportError as exc:
            logger.debug('Tray dependencies unavailable: %s', exc)
            return False

        image = Image.new('RGB', (64, 64), color='black')
        draw = ImageDraw.Draw(image)
        draw.rectangle((12, 12, 52, 52), outline='cyan', width=3)
        draw.text((20, 22), 'J', fill='cyan')
        self._icon = pystray.Icon('jarvis', image, self.tooltip)
        self._icon.title = f'{self.tooltip}: {self.status}'
        self._icon.run_detached()
        return True

    def set_status(self, status: str) -> None:
        self.status = status
        if self._icon is not None:
            self._icon.title = f'{self.tooltip}: {self.status}'

    def stop(self) -> None:
        if self._icon is not None:
            self._icon.stop()
        self._icon = None
