from __future__ import annotations

from typing import Any, Dict

from config.config import CONFIG
from core.action_router import ActionRouter
from core.adaptive_agent import AdaptiveAgent
from core.browser_executor import BrowserExecutor
from core.intent_classifier import IntentResult
from core.persona_manager import PersonaManager
from core.self_healer import SelfHealer
from core.vision_engine import VisionEngine
from ui.hud_state_manager import HUDStateManager


class JarvisOrchestrator:
    """v3 orchestrator that layers new systems over the existing adaptive agent."""

    def __init__(self, adaptive_agent: AdaptiveAgent | None = None) -> None:
        self.agent = adaptive_agent or AdaptiveAgent()
        self.hud = HUDStateManager()
        self.vision = VisionEngine(
            enabled=CONFIG.get('VISION_ENABLED', False),
            api_key=CONFIG.get('NVIDIA_API_KEY', ''),
            model=CONFIG.get('VISION_MODEL', 'llama-3.2-90b-vision-instruct'),
        )
        self.browser_executor = BrowserExecutor(
            enabled=CONFIG.get('BROWSER_AUTOMATION_ENABLED', False),
            use_playwright=CONFIG.get('USE_PLAYWRIGHT', False),
        )
        self.self_healer = SelfHealer()
        self.personas = PersonaManager()
        self.router = ActionRouter(confirmation_mode=CONFIG.get('CONFIRMATION_MODE', False))
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.router.register_handler('browser_task', self._handle_browser_task)
        self.router.register_handler('general', self._handle_general)
        self.router.register_handler('open_app', self._handle_general)
        self.router.register_handler('close_app', self._handle_general)
        self.router.register_handler('persona_management', self._handle_persona)

    def process(self, command: str) -> Dict[str, Any]:
        self.hud.set_state('thinking', f'Routing: {command[:80]}')

        def _run() -> Dict[str, Any]:
            routed = self.router.route(command)
            if routed.get('status') == 'needs_confirmation':
                self.hud.set_state('confirm', routed.get('message', 'Confirmation required'))
                return routed
            self.hud.set_state('success', 'Action complete')
            return routed

        recovered = self.self_healer.recover(executor=_run, max_attempts=2)
        if recovered.get('ok'):
            result = recovered.get('result', {})
            if isinstance(result, dict):
                result['self_healing'] = {'attempts': recovered.get('attempts', [])}
                return result
            return {'status': 'ok', 'result': result}
        self.hud.set_state('error', 'Recovery failed')
        return {'status': 'error', 'message': 'Recovery failed', 'attempts': recovered.get('attempts', [])}

    def _handle_general(
        self,
        command: str,
        intent: IntentResult,
        context: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        result = self.agent.process_command(command)
        result.update({'status': 'ok', 'intent': intent.intent})
        return result

    def _handle_browser_task(
        self,
        command: str,
        intent: IntentResult,
        context: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        response = self.browser_executor.run_task(
            url='https://www.google.com',
            action='type',
            target='Search',
            value=command,
        )
        if response.get('status') == 'ok' and CONFIG.get('VISION_ENABLED', False):
            verification = self.vision.verify_goal('Search results are visible')
            response['verification'] = verification
        return response

    def _handle_persona(
        self,
        command: str,
        intent: IntentResult,
        context: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        if 'list' in command.lower():
            return {'status': 'ok', 'personas': self.personas.list_personas()}
        return {'status': 'ok', 'active_persona': self.personas.get_active_persona()}
