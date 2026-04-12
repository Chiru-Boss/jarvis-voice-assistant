"""Pattern Memory – persist and retrieve user interaction patterns.

Stores usage patterns in ``data/user_patterns.json`` and provides fast
in-memory access.  Data is automatically saved after each update to ensure
persistence across sessions.

Pattern categories
------------------
* ``apps``        – frequency, session counts, time-of-day usage.
* ``searches``    – search term frequency.
* ``workflows``   – action sequences with frequency and success rate.
* ``time_patterns`` – which apps / commands are used at morning / afternoon / evening / night.
* ``context``     – most recently active app and last few commands.
* ``command_history`` – full chronological command log.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default path for the pattern database.
_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'user_patterns.json',
)

# Maximum command history entries to keep on disk.
_MAX_COMMAND_HISTORY = 500


def _time_of_day() -> str:
    """Return 'morning', 'afternoon', 'evening', or 'night' for the current time."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return 'morning'
    if 12 <= hour < 17:
        return 'afternoon'
    if 17 <= hour < 21:
        return 'evening'
    return 'night'


class PatternMemory:
    """Load, update, and persist user interaction patterns.

    Parameters
    ----------
    db_path : str
        Path to the JSON pattern database file.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        """Load patterns from disk, returning a blank structure on failure."""
        blank: Dict[str, Any] = {
            'apps': {},
            'searches': {},
            'workflows': [],
            'time_patterns': {'morning': [], 'afternoon': [], 'evening': [], 'night': []},
            'context': {'current_app': '', 'recent_commands': [], 'user_preferences': {}},
            'command_history': [],
            'last_updated': None,
        }
        if not os.path.exists(self._db_path):
            return blank
        try:
            with open(self._db_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            # Fill in any keys that might be missing from older files.
            for key, value in blank.items():
                data.setdefault(key, value)
            return data
        except Exception as exc:
            logger.warning('Pattern DB load failed (%s) – starting fresh.', exc)
            return blank

    def save(self) -> None:
        """Persist current patterns to disk."""
        self._data['last_updated'] = datetime.utcnow().isoformat()
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            with open(self._db_path, 'w', encoding='utf-8') as fh:
                json.dump(self._data, fh, indent=2)
        except Exception as exc:
            logger.error('Pattern DB save failed: %s', exc)

    # ------------------------------------------------------------------
    # App patterns
    # ------------------------------------------------------------------

    def record_app_open(self, app_name: str) -> None:
        """Record that *app_name* was opened."""
        app_key = app_name.lower()
        apps = self._data['apps']
        tod = _time_of_day()

        if app_key not in apps:
            apps[app_key] = {
                'frequency': 0,
                'total_sessions': 0,
                'avg_session_duration': 0.0,
                'time_of_day': [],
                'last_used': None,
                '_session_start': None,
            }

        entry = apps[app_key]
        entry['frequency'] += 1
        entry['total_sessions'] += 1
        entry['last_used'] = datetime.utcnow().isoformat()
        entry['_session_start'] = time.time()

        if tod not in entry['time_of_day']:
            entry['time_of_day'].append(tod)

        # Update time_patterns.
        tp = self._data['time_patterns'][tod]
        if app_key not in tp:
            tp.append(app_key)

        self.save()

    def record_app_close(self, app_name: str) -> None:
        """Record that *app_name* was closed and update session duration."""
        app_key = app_name.lower()
        apps = self._data['apps']
        if app_key not in apps:
            return

        entry = apps[app_key]
        start = entry.get('_session_start')
        if start:
            duration = time.time() - start
            prev_avg = entry['avg_session_duration']
            n = entry['total_sessions']
            # Rolling average.
            entry['avg_session_duration'] = prev_avg + (duration - prev_avg) / n
            entry['_session_start'] = None

        self.save()

    def get_top_apps(self, n: int = 5) -> List[str]:
        """Return the *n* most frequently used app names."""
        apps = self._data['apps']
        sorted_apps = sorted(apps.keys(), key=lambda k: apps[k].get('frequency', 0), reverse=True)
        return sorted_apps[:n]

    # ------------------------------------------------------------------
    # Search patterns
    # ------------------------------------------------------------------

    def record_search(self, term: str) -> None:
        """Record a search term."""
        key = term.lower().strip()
        searches = self._data['searches']
        searches[key] = searches.get(key, 0) + 1
        self.save()

    def get_top_searches(self, n: int = 5) -> List[str]:
        """Return the *n* most frequent search terms."""
        searches = self._data['searches']
        return sorted(searches.keys(), key=lambda k: searches[k], reverse=True)[:n]

    # ------------------------------------------------------------------
    # Workflow sequences
    # ------------------------------------------------------------------

    def record_workflow(self, sequence: List[str], success: bool = True) -> None:
        """Record a multi-step workflow sequence."""
        workflows: List[Dict[str, Any]] = self._data['workflows']
        # Look for an existing matching sequence.
        for wf in workflows:
            if wf.get('sequence') == sequence:
                wf['frequency'] = wf.get('frequency', 0) + 1
                n = wf['frequency']
                prev_rate = wf.get('success_rate', 1.0)
                wf['success_rate'] = prev_rate + (int(success) - prev_rate) / n
                self.save()
                return

        workflows.append({
            'sequence': sequence,
            'frequency': 1,
            'success_rate': 1.0 if success else 0.0,
        })
        self.save()

    def get_top_workflows(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the *n* most frequent workflows."""
        wfs = self._data['workflows']
        return sorted(wfs, key=lambda w: w.get('frequency', 0), reverse=True)[:n]

    # ------------------------------------------------------------------
    # Command history
    # ------------------------------------------------------------------

    def record_command(self, command: str) -> None:
        """Append *command* to the command history."""
        history: List[Dict[str, Any]] = self._data['command_history']
        history.append({
            'command': command,
            'timestamp': datetime.utcnow().isoformat(),
            'time_of_day': _time_of_day(),
        })
        # Trim to avoid unbounded growth.
        if len(history) > _MAX_COMMAND_HISTORY:
            self._data['command_history'] = history[-_MAX_COMMAND_HISTORY:]
        self._update_recent_commands(command)
        self.save()

    def get_command_history(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Return the last *last_n* command history entries."""
        return self._data['command_history'][-last_n:]

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def set_current_app(self, app_name: str) -> None:
        """Update the currently active application in context."""
        self._data['context']['current_app'] = app_name
        self.save()

    def get_current_app(self) -> str:
        """Return the most recently active application."""
        return self._data['context'].get('current_app', '')

    def get_recent_commands(self, n: int = 5) -> List[str]:
        """Return the *n* most recent command strings."""
        return self._data['context'].get('recent_commands', [])[-n:]

    def get_time_patterns(self, time_of_day: Optional[str] = None) -> List[str]:
        """Return app names associated with *time_of_day* (default: current)."""
        tod = time_of_day or _time_of_day()
        return self._data['time_patterns'].get(tod, [])

    def set_preference(self, key: str, value: Any) -> None:
        """Store a user preference."""
        self._data['context']['user_preferences'][key] = value
        self.save()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve a user preference."""
        return self._data['context']['user_preferences'].get(key, default)

    def get_all_patterns(self) -> Dict[str, Any]:
        """Return a snapshot of all pattern data (for MCP tool)."""
        snapshot = {
            'top_apps': self.get_top_apps(10),
            'top_searches': self.get_top_searches(10),
            'top_workflows': self.get_top_workflows(5),
            'current_time_of_day': _time_of_day(),
            'time_pattern_apps': self.get_time_patterns(),
            'recent_commands': self.get_recent_commands(10),
            'current_app': self.get_current_app(),
        }
        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_recent_commands(self, command: str) -> None:
        recent: List[str] = self._data['context'].get('recent_commands', [])
        recent.append(command)
        self._data['context']['recent_commands'] = recent[-20:]
