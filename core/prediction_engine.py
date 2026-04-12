"""Prediction Engine – anticipate the user's next action.

Uses data from :class:`~core.behavior_learner.BehaviorLearner` to generate
ranked suggestions for what JARVIS should offer or prepare next.

Prediction strategies
---------------------
1. **Sequence continuation** – if the last command matches the start of a
   known bigram / workflow, suggest the follow-up.
2. **Time-based** – apps / commands the user typically runs at this time
   of day.
3. **Frequency fallback** – the globally most-used app / command when no
   stronger signal exists.

All suggestions carry a ``confidence`` score in [0, 1].
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.behavior_learner import BehaviorLearner
from core.pattern_memory import PatternMemory, _time_of_day

logger = logging.getLogger(__name__)

# Minimum confidence below which a suggestion is suppressed.
_MIN_CONFIDENCE_THRESHOLD = 0.2


class PredictionEngine:
    """Generate next-action predictions for JARVIS.

    Parameters
    ----------
    memory : PatternMemory
        The shared pattern memory instance.
    learner : BehaviorLearner
        The behaviour analysis layer.
    """

    def __init__(self, memory: PatternMemory, learner: BehaviorLearner) -> None:
        self._memory = memory
        self._learner = learner

    # ------------------------------------------------------------------
    # Primary prediction method
    # ------------------------------------------------------------------

    def predict_next(
        self,
        last_command: Optional[str] = None,
        max_suggestions: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return ranked action suggestions.

        Parameters
        ----------
        last_command : str or None
            The most recent command, used for sequence-based prediction.
        max_suggestions : int
            Maximum number of suggestions to return.

        Returns
        -------
        list[dict]
            Each dict has ``action``, ``confidence``, and ``reason`` keys.
        """
        candidates: List[Dict[str, Any]] = []

        # 1. Sequence-based prediction.
        if last_command:
            candidates.extend(self._sequence_predictions(last_command))

        # 2. Time-of-day prediction.
        candidates.extend(self._time_predictions())

        # 3. Frequency fallback.
        candidates.extend(self._frequency_predictions())

        # Deduplicate (keep highest confidence for each action).
        seen: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            key = c['action'].lower()
            if key not in seen or c['confidence'] > seen[key]['confidence']:
                seen[key] = c

        # Filter low-confidence and sort descending.
        sorted_cands = sorted(
            [c for c in seen.values() if c['confidence'] >= _MIN_CONFIDENCE_THRESHOLD],
            key=lambda x: x['confidence'],
            reverse=True,
        )
        return sorted_cands[:max_suggestions]

    def predict_action_text(
        self,
        last_command: Optional[str] = None,
        max_suggestions: int = 3,
    ) -> str:
        """Return a human-readable prediction string (for MCP tool use)."""
        suggestions = self.predict_next(last_command=last_command, max_suggestions=max_suggestions)
        if not suggestions:
            return 'No predictions available yet. Keep using JARVIS to build patterns!'

        lines = ['🔮 Predicted next actions:']
        for i, s in enumerate(suggestions, 1):
            pct = int(s['confidence'] * 100)
            lines.append(f"  {i}. {s['action']} ({pct}% confidence) – {s['reason']}")
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Prediction strategies
    # ------------------------------------------------------------------

    def _sequence_predictions(self, last_command: str) -> List[Dict[str, Any]]:
        """Predict based on known command sequences following *last_command*."""
        results = []
        sequences = self._learner.detect_sequences(min_frequency=1)
        cmd_lower = last_command.lower()

        for seq_info in sequences:
            seq = seq_info.get('sequence', [])
            if len(seq) >= 2 and seq[0].lower() == cmd_lower:
                results.append({
                    'action': seq[1],
                    'confidence': seq_info.get('confidence', 0.3),
                    'reason': f"Follows '{last_command}' in {seq_info.get('frequency', 1)} past sessions",
                })
        return results

    def _time_predictions(self) -> List[Dict[str, Any]]:
        """Suggest apps that the user typically opens at this time of day."""
        tod = _time_of_day()
        tod_apps = self._memory.get_time_patterns(tod)
        results = []
        app_data = self._memory._data.get('apps', {})

        for app_name in tod_apps[:5]:
            freq = app_data.get(app_name, {}).get('frequency', 1)
            confidence = min(0.8, 0.3 + freq * 0.05)
            results.append({
                'action': f'open {app_name}',
                'confidence': round(confidence, 2),
                'reason': f'You usually open this app in the {tod}',
            })
        return results

    def _frequency_predictions(self) -> List[Dict[str, Any]]:
        """Suggest the user's most-used apps as a frequency fallback."""
        top_apps = self._learner.most_used_apps(3)
        results = []
        for item in top_apps:
            confidence = min(0.6, item.get('confidence', 0.1) * 0.5)
            results.append({
                'action': f"open {item['app']}",
                'confidence': round(confidence, 2),
                'reason': f"Frequently used ({item['frequency']} times)",
            })
        return results
