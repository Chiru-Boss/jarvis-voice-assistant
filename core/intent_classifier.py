"""Intent classification helpers for JARVIS v3 routing."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float


class IntentClassifier:
    """Lightweight regex-based intent classifier.

    Keeps routing deterministic and dependency-free while allowing optional
    executor-specific logic to be plugged in by higher layers.
    """

    _PATTERNS = {
        'persona_switch': re.compile(r'\b(?:switch persona|use persona|load persona)\b', re.I),
        'goal_verification': re.compile(r'\b(?:was .*sent|verify|confirm|did .*work|check if)\b', re.I),
        'browser_automation': re.compile(r'\b(?:website|browser|click|form|scroll|navigate|book|checkout)\b', re.I),
        'system_command': re.compile(r'\b(?:run command|terminal|execute)\b', re.I),
    }

    def classify(self, text: str) -> IntentResult:
        text = (text or '').strip()
        if not text:
            return IntentResult(intent='general', confidence=0.2)

        for name, pattern in self._PATTERNS.items():
            if pattern.search(text):
                return IntentResult(intent=name, confidence=0.85)

        return IntentResult(intent='general', confidence=0.55)
