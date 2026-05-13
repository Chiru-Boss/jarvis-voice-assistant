from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class VisionEngine:
    """NVIDIA NIM vision integration for screenshot analysis and verification."""

    api_url = 'https://integrate.api.nvidia.com/v1/chat/completions'

    def __init__(
        self,
        *,
        api_key: str = '',
        model: str = 'llama-3.2-90b-vision-instruct',
        enabled: bool = False,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.enabled = enabled
        self.timeout = timeout

    def analyze_screenshot(
        self,
        *,
        goal: str = '',
        image_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {'ok': False, 'error': 'Vision engine disabled.'}
        if not self.api_key:
            return {'ok': False, 'error': 'NVIDIA_API_KEY is not configured.'}

        data = image_bytes if image_bytes is not None else self.capture_screenshot_bytes()
        if not data:
            return {'ok': False, 'error': 'Screenshot capture failed.'}

        prompt = (
            'You are a desktop automation verifier. Analyze the screenshot and describe '
            'the current UI state concisely.'
        )
        if goal:
            prompt += (
                f" Then answer if this goal is achieved: '{goal}'. "
                "Return 'GOAL_VERIFIED: yes' or 'GOAL_VERIFIED: no'."
            )

        content = self._call_vision_model(image_bytes=data, prompt=prompt)
        goal_verified = 'goal_verified: yes' in content.lower()
        return {
            'ok': True,
            'model': self.model,
            'analysis': content,
            'goal_verified': goal_verified,
        }

    def verify_goal(self, goal: str, *, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        return self.analyze_screenshot(goal=goal, image_bytes=image_bytes)

    def capture_screenshot_bytes(self) -> Optional[bytes]:
        try:
            from PIL import ImageGrab  # type: ignore
        except ImportError as exc:
            logger.debug('Pillow ImageGrab unavailable: %s', exc)
            return None
        try:
            img = ImageGrab.grab()
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()
        except OSError as exc:
            logger.debug('Screenshot capture failed: %s', exc)
            return None
        except ValueError as exc:
            logger.debug('Screenshot encoding failed: %s', exc)
            return None

    def _call_vision_model(self, *, image_bytes: bytes, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        payload = {
            'model': self.model,
            'temperature': 0.2,
            'max_tokens': 512,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/png;base64,{b64}'},
                        },
                    ],
                }
            ],
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
        }
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return body.get('choices', [{}])[0].get('message', {}).get('content', '')
