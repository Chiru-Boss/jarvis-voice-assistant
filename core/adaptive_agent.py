"""Adaptive Agent – orchestrate all JARVIS subsystems into a unified AI agent.

This is the central coordinator that:

1. Accepts a user command (text).
2. Queries the screen vision for context.
3. Consults the behaviour learner for patterns.
4. Routes the command to the appropriate executor (AppController / SystemExecutor).
5. Learns from the outcome.
6. Returns the result and any predictions for the next action.

The hand-tracking mouse (hologram cursor) is **not** managed here – it runs in
its own thread via :class:`~core.hand_voice_integration.HandVoiceIntegration`
and is completely independent of this agent.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from core.app_controller import AppController
from core.behavior_learner import BehaviorLearner
from core.pattern_memory import PatternMemory
from core.prediction_engine import PredictionEngine
from core.screen_vision import ScreenVision
from core.system_executor import SystemExecutor

logger = logging.getLogger(__name__)

# Simple keyword patterns to extract intent from a command string.
_APP_OPEN_PATTERNS = re.compile(
    r'\b(?:open|launch|start|run)\s+([a-zA-Z0-9_\-. ]+)',
    re.IGNORECASE,
)
_APP_CLOSE_PATTERNS = re.compile(
    r'\b(?:close|quit|exit|kill|stop)\s+([a-zA-Z0-9_\-. ]+)',
    re.IGNORECASE,
)
_SEARCH_PATTERNS = re.compile(
    r'\b(?:search|find|look up|google|browse)\s+(?:for\s+)?(.+)',
    re.IGNORECASE,
)
_FILE_READ_PATTERNS = re.compile(
    r'\b(?:read|open|show|cat|display)\s+(?:file\s+)?["\']?([^\s"\']+)["\']?',
    re.IGNORECASE,
)
_CMD_PATTERNS = re.compile(
    r'\b(?:run command|execute|cmd|terminal)\s+["\']?(.+)["\']?',
    re.IGNORECASE,
)


class AdaptiveAgent:
    """Orchestrate all JARVIS subsystems.

    Parameters
    ----------
    pattern_db_path : str or None
        Override path for the pattern database (uses default if None).
    """

    def __init__(self, pattern_db_path: Optional[str] = None) -> None:
        kwargs = {'db_path': pattern_db_path} if pattern_db_path else {}
        self._memory = PatternMemory(**kwargs)
        self._learner = BehaviorLearner(self._memory)
        self._predictor = PredictionEngine(self._memory, self._learner)
        self._vision = ScreenVision()
        self._app_ctrl = AppController()
        self._executor = SystemExecutor()

        # Rolling workflow tracking: accumulate recent commands.
        self._recent_workflow: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def memory(self) -> PatternMemory:
        return self._memory

    @property
    def learner(self) -> BehaviorLearner:
        return self._learner

    @property
    def predictor(self) -> PredictionEngine:
        return self._predictor

    @property
    def vision(self) -> ScreenVision:
        return self._vision

    @property
    def app_controller(self) -> AppController:
        return self._app_ctrl

    @property
    def executor(self) -> SystemExecutor:
        return self._executor

    def process_command(self, command: str) -> Dict[str, Any]:
        """Process *command* and return a result dict.

        Keys in the returned dict
        -------------------------
        * ``result`` – str action result.
        * ``predictions`` – list of next-action predictions.
        * ``screen_context`` – brief screen summary (or empty string).
        * ``patterns_applied`` – bool, True if patterns were used.
        """
        start = time.monotonic()
        logger.info('AdaptiveAgent: processing command: %s', command)

        result_text = ''
        patterns_applied = False
        app_opened: Optional[str] = None
        search_term: Optional[str] = None

        # ── 1. App launch intent ─────────────────────────────────────────
        open_match = _APP_OPEN_PATTERNS.search(command)
        if open_match:
            target_app = open_match.group(1).strip().rstrip('.')
            result_text = self._app_ctrl.open_app(target_app)
            app_opened = target_app

        # ── 2. App close intent ──────────────────────────────────────────
        close_match = _APP_CLOSE_PATTERNS.search(command)
        if close_match and not open_match:
            target_app = close_match.group(1).strip().rstrip('.')
            result_text = self._app_ctrl.close_app(target_app)
            self._memory.record_app_close(target_app)

        # ── 3. Search intent ─────────────────────────────────────────────
        search_match = _SEARCH_PATTERNS.search(command)
        if search_match:
            search_term = search_match.group(1).strip()
            self._memory.record_search(search_term)
            if not result_text:
                result_text = f"📝 Search term noted: '{search_term}'."

        # ── 4. Shell command intent ──────────────────────────────────────
        cmd_match = _CMD_PATTERNS.search(command)
        if cmd_match and not result_text:
            shell_cmd = cmd_match.group(1).strip().strip('"\'')
            result_text = self._executor.execute_command(shell_cmd)

        # ── 5. Learn from this interaction ───────────────────────────────
        self._recent_workflow.append(command)
        if len(self._recent_workflow) > 10:
            self._recent_workflow = self._recent_workflow[-10:]

        self._learner.learn_from_interaction(
            command=command,
            app_opened=app_opened,
            search_term=search_term,
            workflow_sequence=self._recent_workflow[-3:] if len(self._recent_workflow) >= 2 else None,
            success=True,
        )

        # ── 6. Get screen context (lightweight) ──────────────────────────
        screen_summary = self._get_screen_summary()

        # ── 7. Predict next action ───────────────────────────────────────
        predictions = self._predictor.predict_next(
            last_command=command,
            max_suggestions=3,
        )

        elapsed = time.monotonic() - start
        logger.debug('AdaptiveAgent: processed in %.2f s', elapsed)

        return {
            'result': result_text or '',
            'predictions': predictions,
            'screen_context': screen_summary,
            'patterns_applied': patterns_applied,
        }

    def get_pattern_summary(self) -> str:
        """Return a string summary of learned patterns for the LLM system prompt."""
        summary = self._learner.get_behavioral_summary()
        top_apps = [a['app'] for a in summary.get('top_apps', [])[:5]]
        top_searches = [s['term'] for s in summary.get('top_searches', [])[:5]]
        tod_apps = summary.get('time_patterns', {}).get('expected_apps', [])

        parts = []
        if top_apps:
            parts.append(f"Most-used apps: {', '.join(top_apps)}")
        if top_searches:
            parts.append(f"Common searches: {', '.join(top_searches)}")
        if tod_apps:
            parts.append(f"Typical apps at this time of day: {', '.join(tod_apps[:3])}")

        return '; '.join(parts) if parts else ''

    def learn_from_gesture(self, gesture_name: str) -> None:
        """Record a hand gesture as a pattern interaction (non-command)."""
        if gesture_name and gesture_name not in ('none', 'point', ''):
            self._memory.record_command(f'[gesture] {gesture_name}')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_screen_summary(self) -> str:
        """Return a brief one-line summary of the current screen."""
        try:
            content = self._vision.get_screen_content(force=False)
            if not content.get('captured'):
                return ''
            ocr = content.get('ocr_text', '')
            if not ocr:
                return ''
            # Return first 200 chars of visible text.
            brief = ' '.join(ocr.split())[:200]
            return f"[Screen: {brief}]"
        except Exception as exc:
            logger.debug('Screen summary failed: %s', exc)
            return ''
