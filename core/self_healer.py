from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class SelfHealer:
    """Autonomous failure diagnosis and recovery planning."""

    def diagnose_failure(self, *, error: str, step: str = '') -> Dict[str, str]:
        lowered = error.lower()
        if 'timeout' in lowered:
            return {
                'category': 'timeout',
                'reason': error,
                'recommendation': 'Increase timeout and retry with stability checks.',
                'step': step,
            }
        if 'selector' in lowered or 'element' in lowered:
            return {
                'category': 'selector',
                'reason': error,
                'recommendation': 'Use semantic selector fallback and scroll for element.',
                'step': step,
            }
        if 'permission' in lowered or 'denied' in lowered:
            return {
                'category': 'permissions',
                'reason': error,
                'recommendation': 'Request user confirmation or elevated permissions.',
                'step': step,
            }
        return {
            'category': 'general',
            'reason': error,
            'recommendation': 'Re-plan action with simpler steps and verify goal.',
            'step': step,
        }

    def build_recovery_plan(self, diagnosis: Dict[str, str]) -> List[str]:
        category = diagnosis.get('category', 'general')
        if category == 'timeout':
            return ['refresh_context', 'increase_timeout', 'retry']
        if category == 'selector':
            return ['load_cached_selector', 'semantic_selector_search', 'smart_scroll', 'retry']
        if category == 'permissions':
            return ['request_confirmation', 'retry_with_confirmation']
        return ['replan', 'retry']

    def recover(
        self,
        *,
        executor: Callable[[], Any],
        max_attempts: int = 2,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = context or {}
        attempts: List[Dict[str, str]] = []
        for attempt in range(1, max_attempts + 1):
            try:
                result = executor()
                return {
                    'ok': True,
                    'attempt': attempt,
                    'result': result,
                    'attempts': attempts,
                    'context': ctx,
                }
            except Exception as exc:
                diagnosis = self.diagnose_failure(error=str(exc), step=f'attempt_{attempt}')
                plan = self.build_recovery_plan(diagnosis)
                attempts.append({'diagnosis': diagnosis['category'], 'plan': ' -> '.join(plan)})
        return {'ok': False, 'attempt': max_attempts, 'attempts': attempts, 'context': ctx}
