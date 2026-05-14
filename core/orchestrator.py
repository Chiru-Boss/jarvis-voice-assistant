"""Unified optional orchestrator for JARVIS v3 modules."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from core.action_router import ActionRouter
from core.adaptive_agent import AdaptiveAgent
from core.browser_executor import BrowserAutomationExecutor
from core.persona_manager import PersonaManager
from core.self_healer import SelfHealer
from core.vision_engine import VisionEngine
from ui.overlay import OverlayHUD
from ui.tray import TrayIndicator


class Orchestrator:
    """Coordinate optional v3 modules while preserving classic behavior."""

    def __init__(self, agent: Optional[AdaptiveAgent] = None):
        def _safe_float(value: str, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        self.agent = agent or AdaptiveAgent()
        self.router = ActionRouter()
        self.vision = VisionEngine()
        self.browser = BrowserAutomationExecutor()
        self.self_healer = SelfHealer(max_attempts=int(os.getenv('MAX_RECOVERY_ATTEMPTS', '3')))
        self.persona_manager = PersonaManager(data_dir=os.getenv('PERSONA_DATA_DIR', 'data/personas'))
        self.overlay = OverlayHUD(
            enabled=os.getenv('ENABLE_OVERLAY_UI', 'false').lower() == 'true',
            position=os.getenv('OVERLAY_POSITION', 'top-right'),
            opacity=_safe_float(os.getenv('OVERLAY_OPACITY', '0.9'), 0.9),
        )
        self.tray = TrayIndicator(enabled=self.overlay.enabled)

    def process(self, command: str, *, screenshot_bytes: bytes = b'', page_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a command through optional v3 routing.

        - Uses action routing to select vision, browser, persona, or fallback
          adaptive-agent execution.
        - *screenshot_bytes* is used by vision verification routes.
        - *page_snapshot* is used by browser semantic targeting routes.
        Returns ``{'route': ..., 'result': ...}`` on success, or includes
        ``error`` and ``recovery`` fields on failures.
        """
        caps = {
            'vision': self.vision.enabled,
            'browser': self.browser.enabled,
            'persona_system': os.getenv('PERSONA_SYSTEM_ENABLED', 'false').lower() == 'true',
        }
        decision = self.router.decide(command, capabilities=caps)

        self.overlay.update('thinking', f'Route: {decision.route}')
        self.tray.set_status('thinking', decision.reason)

        try:
            if decision.route == 'vision':
                self.overlay.update('executing', 'Verifying task goal with vision')
                result = self.vision.verify_goal(command, image_bytes=screenshot_bytes)
                self.overlay.update('success' if result.get('completed') else 'error', 'Vision verification complete')
                return {'route': decision.route, 'result': result}

            if decision.route == 'browser':
                self.overlay.update('executing', 'Running browser automation')
                result = self.browser.execute_task(command, page_snapshot=page_snapshot)
                self.overlay.update('success' if result.get('ok') else 'error', result.get('message', ''))
                return {'route': decision.route, 'result': result}

            if decision.route == 'persona':
                name = self._extract_persona_name(command)
                persona = self.persona_manager.switch_persona(name)
                self.overlay.update('success', f'Persona switched to {name}')
                return {'route': decision.route, 'result': persona}

            # Fallback to existing adaptive agent behavior.
            self.overlay.update('executing', 'Using adaptive agent')
            result = self.agent.process_command(command)
            self.overlay.update('success', 'Adaptive task complete')
            return {'route': decision.route, 'result': result}
        except Exception as exc:
            healed = self.self_healer.recover(command, str(exc), attempt_fn=lambda: False)
            self.overlay.update('error', healed.diagnosis)
            return {
                'route': decision.route,
                'error': str(exc),
                'recovery': {
                    'diagnosis': healed.diagnosis,
                    'plan': healed.plan,
                    'attempts': healed.attempts,
                    'recovered': healed.recovered,
                },
            }

    @staticmethod
    def _extract_persona_name(command: str) -> str:
        text = (command or '').strip()
        match = re.search(r'(?:switch persona|use persona|load persona)\s+(.+)$', text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
        return text.split()[-1].strip().lower() if text else ''
