from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: Dict[str, str] = field(default_factory=dict)
    requires_confirmation: bool = False


class IntentClassifier:
    """Lightweight intent classifier used by the v3 action router."""

    _SEARCH = re.compile(r'\b(?:search|find|look up|google|browse)\b', re.IGNORECASE)
    _OPEN = re.compile(r'\b(?:open|launch|start|run)\s+(.+)', re.IGNORECASE)
    _CLOSE = re.compile(r'\b(?:close|quit|exit|kill|stop)\s+(.+)', re.IGNORECASE)
    _PERSONA = re.compile(r'\bpersona\b', re.IGNORECASE)

    def classify(self, command: str) -> IntentResult:
        text = command.strip()
        lowered = text.lower()

        if self._PERSONA.search(text):
            return IntentResult(intent='persona_management', confidence=0.8)

        open_match = self._OPEN.search(text)
        if open_match:
            return IntentResult(
                intent='open_app',
                confidence=0.85,
                entities={'app': open_match.group(1).strip()},
            )

        close_match = self._CLOSE.search(text)
        if close_match:
            return IntentResult(
                intent='close_app',
                confidence=0.82,
                entities={'app': close_match.group(1).strip()},
                requires_confirmation=True,
            )

        if self._SEARCH.search(text):
            return IntentResult(intent='browser_task', confidence=0.88, entities={'query': text})

        if any(keyword in lowered for keyword in ('delete', 'shutdown', 'reboot', 'format', 'wipe')):
            return IntentResult(
                intent='sensitive_action',
                confidence=0.75,
                requires_confirmation=True,
            )

        return IntentResult(intent='general', confidence=0.55)
