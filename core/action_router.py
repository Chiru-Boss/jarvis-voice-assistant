"""Route commands to the right executor domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from core.intent_classifier import IntentClassifier


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str


class ActionRouter:
    """Map intents to coarse-grained execution routes."""

    def __init__(self, classifier: Optional[IntentClassifier] = None):
        self._classifier = classifier or IntentClassifier()

    def decide(self, command: str, *, capabilities: Optional[Dict[str, bool]] = None) -> RouteDecision:
        capabilities = capabilities or {}
        intent = self._classifier.classify(command)

        if intent.intent == 'persona_switch' and capabilities.get('persona_system', False):
            return RouteDecision(route='persona', reason='persona switch intent')

        if intent.intent == 'goal_verification' and capabilities.get('vision', False):
            return RouteDecision(route='vision', reason='goal verification intent')

        if intent.intent == 'browser_automation' and capabilities.get('browser', False):
            return RouteDecision(route='browser', reason='browser automation intent')

        if intent.intent == 'system_command':
            return RouteDecision(route='system', reason='system command intent')

        return RouteDecision(route='adaptive_agent', reason='default adaptive route')
