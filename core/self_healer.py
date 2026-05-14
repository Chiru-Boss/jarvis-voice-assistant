"""Self-healing logic for failed automation steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class RecoveryResult:
    recovered: bool
    attempts: int
    diagnosis: str
    plan: List[str]


class SelfHealer:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max(1, int(max_attempts))

    def diagnose(self, action: str, error: str) -> str:
        return f"The action '{action}' failed because: {error}. I'll try a safer recovery path."

    def build_plan(self, action: str) -> List[str]:
        return [
            'Re-scan the current page/screen state',
            f"Re-locate the target for '{action}' using semantic matching",
            'Retry with stability waits for UI/network completion',
        ]

    def recover(self, action: str, error: str, attempt_fn: Callable[[], bool]) -> RecoveryResult:
        diagnosis = self.diagnose(action, error)
        plan = self.build_plan(action)
        attempts = 0
        for _ in range(self.max_attempts):
            attempts += 1
            if attempt_fn():
                return RecoveryResult(recovered=True, attempts=attempts, diagnosis=diagnosis, plan=plan)
        return RecoveryResult(recovered=False, attempts=attempts, diagnosis=diagnosis, plan=plan)
