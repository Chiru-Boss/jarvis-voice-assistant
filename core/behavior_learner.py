"""Behavior Learner – analyse user patterns to produce actionable insights.

This module processes raw data from :class:`~core.pattern_memory.PatternMemory`
and extracts higher-level behavioral insights:

* **Frequency analysis** – which apps / searches / commands occur most often.
* **Sequence detection** – which commands tend to follow each other.
* **Time pattern recognition** – what the user does at different times of day.
* **Context associations** – what actions co-occur with open applications.
* **Confidence scoring** – how reliable each insight is.

All methods are stateless with respect to file I/O – the PatternMemory object
is the single source of truth.  Insights are returned as plain dicts so they
can be serialised and passed to the LLM.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.pattern_memory import PatternMemory, _time_of_day

logger = logging.getLogger(__name__)

# Minimum number of observations required before we trust a pattern.
_MIN_CONFIDENCE_OBSERVATIONS = 3
# Decay factor – patterns older than this many commands count less.
_RECENCY_WINDOW = 50


class BehaviorLearner:
    """Derive insights from :class:`PatternMemory`.

    Parameters
    ----------
    memory : PatternMemory
        The shared pattern memory instance.
    """

    def __init__(self, memory: PatternMemory) -> None:
        self._memory = memory

    # ------------------------------------------------------------------
    # Frequency analysis
    # ------------------------------------------------------------------

    def most_used_apps(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the top-*n* apps with frequency and confidence."""
        apps = self._memory._data.get('apps', {})
        ranked = sorted(apps.items(), key=lambda kv: kv[1].get('frequency', 0), reverse=True)
        result = []
        for name, info in ranked[:n]:
            freq = info.get('frequency', 0)
            confidence = min(1.0, freq / max(_MIN_CONFIDENCE_OBSERVATIONS, 1))
            result.append({
                'app': name,
                'frequency': freq,
                'confidence': round(confidence, 2),
                'avg_session_min': round(info.get('avg_session_duration', 0) / 60, 1),
                'time_of_day': info.get('time_of_day', []),
            })
        return result

    def most_searched_terms(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the top-*n* search terms with frequency and confidence."""
        searches = self._memory._data.get('searches', {})
        ranked = sorted(searches.items(), key=lambda kv: kv[1], reverse=True)
        result = []
        for term, freq in ranked[:n]:
            confidence = min(1.0, freq / max(_MIN_CONFIDENCE_OBSERVATIONS, 1))
            result.append({'term': term, 'frequency': freq, 'confidence': round(confidence, 2)})
        return result

    # ------------------------------------------------------------------
    # Sequence detection
    # ------------------------------------------------------------------

    def detect_sequences(self, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """Find command bigrams that occur together at least *min_frequency* times.

        Uses the command history to build a transition matrix.
        """
        history = self._memory._data.get('command_history', [])
        commands = [entry.get('command', '') for entry in history if entry.get('command')]

        # Build bigram counts.
        bigrams: Counter = Counter()
        for i in range(len(commands) - 1):
            bigrams[(commands[i], commands[i + 1])] += 1

        sequences = []
        total = max(len(commands) - 1, 1)
        for (a, b), count in bigrams.most_common(20):
            if count < min_frequency:
                break
            confidence = min(1.0, count / max(_MIN_CONFIDENCE_OBSERVATIONS, 1))
            probability = round(count / total, 3)
            sequences.append({
                'sequence': [a, b],
                'frequency': count,
                'confidence': round(confidence, 2),
                'probability': probability,
            })
        return sequences

    # ------------------------------------------------------------------
    # Time pattern recognition
    # ------------------------------------------------------------------

    def time_patterns(self) -> Dict[str, Any]:
        """Return a summary of app usage by time of day."""
        tp = self._memory._data.get('time_patterns', {})
        current_tod = _time_of_day()
        return {
            'current_time_of_day': current_tod,
            'expected_apps': tp.get(current_tod, []),
            'all_patterns': tp,
        }

    # ------------------------------------------------------------------
    # Context associations
    # ------------------------------------------------------------------

    def context_associations(self) -> List[Dict[str, Any]]:
        """Detect which commands are commonly run when a given app is open.

        Scans the workflow table for the highest-frequency sequences.
        """
        workflows = self._memory._data.get('workflows', [])
        assoc = []
        for wf in sorted(workflows, key=lambda w: w.get('frequency', 0), reverse=True)[:10]:
            seq = wf.get('sequence', [])
            freq = wf.get('frequency', 0)
            rate = wf.get('success_rate', 1.0)
            if freq >= 2:
                confidence = min(1.0, freq / max(_MIN_CONFIDENCE_OBSERVATIONS, 1))
                assoc.append({
                    'sequence': seq,
                    'frequency': freq,
                    'success_rate': round(rate, 2),
                    'confidence': round(confidence, 2),
                })
        return assoc

    # ------------------------------------------------------------------
    # Summary insight
    # ------------------------------------------------------------------

    def get_behavioral_summary(self) -> Dict[str, Any]:
        """Return a combined insight dict suitable for injecting into the LLM system prompt."""
        return {
            'top_apps': self.most_used_apps(5),
            'top_searches': self.most_searched_terms(5),
            'common_sequences': self.detect_sequences(min_frequency=2),
            'time_patterns': self.time_patterns(),
            'context_associations': self.context_associations(),
        }

    # ------------------------------------------------------------------
    # Learning from a new interaction
    # ------------------------------------------------------------------

    def learn_from_interaction(
        self,
        command: str,
        app_opened: Optional[str] = None,
        search_term: Optional[str] = None,
        workflow_sequence: Optional[List[str]] = None,
        success: bool = True,
    ) -> None:
        """Update patterns based on what just happened.

        This is the primary entry point called after each JARVIS interaction.
        """
        self._memory.record_command(command)

        if app_opened:
            self._memory.record_app_open(app_opened)
            self._memory.set_current_app(app_opened)

        if search_term:
            self._memory.record_search(search_term)

        if workflow_sequence and len(workflow_sequence) >= 2:
            self._memory.record_workflow(workflow_sequence, success=success)
