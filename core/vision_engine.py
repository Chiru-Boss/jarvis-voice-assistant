"""NVIDIA vision integration for screenshot analysis and goal verification."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from utils.vision_cache import VisionCache


@dataclass(frozen=True)
class VisionConfig:
    enabled: bool = False
    model: str = 'llama-3.2-90b-vision-instruct'
    api_key: str = ''
    base_url: str = 'https://integrate.api.nvidia.com/v1'


class VisionEngine:
    """Analyze screenshots and verify goal completion using NVIDIA-hosted vision."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        requester: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: Optional[VisionCache] = None,
    ):
        self._config = config or VisionConfig(
            enabled=os.getenv('VISION_ENABLED', 'false').lower() == 'true',
            model=os.getenv('VISION_MODEL', 'llama-3.2-90b-vision-instruct'),
            api_key=os.getenv('NVIDIA_VISION_API_KEY', ''),
        )
        self._requester = requester or self._default_requester
        self._cache = cache or VisionCache(ttl_seconds=45)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def analyze_screenshot(self, task: str, *, image_bytes: bytes) -> Dict[str, Any]:
        if not self.enabled:
            return {'enabled': False, 'analysis': 'Vision disabled', 'completed': False}
        if not image_bytes:
            return {'enabled': True, 'analysis': 'No screenshot supplied', 'completed': False}
        if not self._config.api_key:
            return {'enabled': True, 'analysis': 'NVIDIA_VISION_API_KEY is missing', 'completed': False}

        cache_key = self._cache.build_key(task, image_bytes)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        payload = {
            'model': self._config.model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': f'Analyze this screenshot and determine if this goal is complete: {task}. Reply with JSON keys: completed, diagnosis, recovery_plan.'},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_b64}'}}
                    ],
                }
            ],
            'temperature': 0.1,
            'max_tokens': 300,
        }

        response = self._requester(payload)
        parsed = self._parse_response(response)
        self._cache.set(cache_key, parsed)
        return parsed

    def verify_goal(self, goal: str, *, image_bytes: bytes) -> Dict[str, Any]:
        return self.analyze_screenshot(goal, image_bytes=image_bytes)

    def diagnose_failure(self, failed_step: str, *, image_bytes: bytes) -> Dict[str, Any]:
        return self.analyze_screenshot(f'Explain why this failed in plain English: {failed_step}', image_bytes=image_bytes)

    def _default_requester(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=self._config.api_key, base_url=self._config.base_url)
        completion = client.chat.completions.create(**payload)
        content = completion.choices[0].message.content if completion.choices else ''
        return {'content': content}

    @staticmethod
    def _parse_response(response: Dict[str, Any]) -> Dict[str, Any]:
        content = response.get('content', '') if isinstance(response, dict) else ''
        if isinstance(content, list):
            content = ' '.join(str(c) for c in content)
        content = str(content or '').strip()

        # Prefer structured JSON when provided.
        try:
            maybe = json.loads(content)
            if isinstance(maybe, dict):
                return {
                    'enabled': True,
                    'completed': bool(maybe.get('completed', False)),
                    'analysis': str(maybe.get('diagnosis', '')),
                    'recovery_plan': maybe.get('recovery_plan', []),
                }
        except Exception:
            pass

        lowered = content.lower()
        completed = '"completed": true' in lowered or 'completed: true' in lowered or 'goal complete' in lowered
        return {
            'enabled': True,
            'completed': completed,
            'analysis': content,
            'recovery_plan': [],
        }
