from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from core.intent_classifier import IntentClassifier, IntentResult

RouteHandler = Callable[[str, IntentResult, Optional[Dict[str, Any]]], Dict[str, Any]]


class ActionRouter:
    """Intent-based dispatcher with confirmation support."""

    def __init__(
        self,
        *,
        classifier: Optional[IntentClassifier] = None,
        confirmation_mode: bool = False,
    ) -> None:
        self.classifier = classifier or IntentClassifier()
        self.confirmation_mode = confirmation_mode
        self._handlers: Dict[str, RouteHandler] = {}

    def register_handler(self, intent: str, handler: RouteHandler) -> None:
        self._handlers[intent] = handler

    def route(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        intent = self.classifier.classify(command)
        if self.confirmation_mode and intent.requires_confirmation:
            return {
                'status': 'needs_confirmation',
                'intent': intent.intent,
                'message': f"Confirmation required for intent '{intent.intent}'.",
                'entities': intent.entities,
            }

        handler = self._handlers.get(intent.intent)
        if handler is None:
            handler = self._handlers.get('general')
        if handler is None:
            return {
                'status': 'unhandled',
                'intent': intent.intent,
                'message': 'No handler registered for this intent.',
                'entities': intent.entities,
            }
        return handler(command, intent, context)
