"""Chat Session Manager – deduplication guard for repetitive confirmation inputs.

Problem context
---------------
When a user repeatedly sends a bare confirmation phrase (e.g.
"@Copilot Accepted Confirmation: Confirm agent session", "yes", "confirmed",
"start it") after a task is already in-flight or has just finished, the
assistant may misinterpret each follow-up acknowledgement as a *new* task
request and queue redundant work (extra PRs, duplicate tool calls, etc.).

This module provides :class:`ChatSessionManager` to:

1. Assign stable IDs to distinct agent work sessions.
2. Detect whether an incoming user message is a bare confirmation/
   acknowledgement with no new actionable intent.
3. Suppress the trigger of a new session when:
   - A session is already ``in_progress``, **or**
   - A session completed within the configurable cooldown window
     (``dedup_window_seconds``).
4. Record session lifecycle events (start, complete) for auditability.

Usage example::

    from core.chat_session import ChatSessionManager

    manager = ChatSessionManager(dedup_window_seconds=60)

    session_id = manager.start_session("feat: add pattern learning")
    # … do work …
    manager.complete_session(session_id)

    # Later the user sends a bare "Confirm agent session" message:
    if manager.is_confirmation_phrase(user_text):
        if manager.should_suppress(user_text):
            # Already handled – acknowledge quietly without re-triggering.
            print("Session already completed. No new work needed.")
        else:
            session_id = manager.start_session(user_text)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confirmation phrase detection
# ---------------------------------------------------------------------------

# Normalised keyword patterns that indicate a bare acknowledgement with no
# new actionable intent.  Phrases that *also* include a substantive request
# (e.g. "confirm and then add dark mode") are intentionally NOT suppressed.
_CONFIRMATION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r'\bconfirm(?:ed|ing|ation)?\b', re.IGNORECASE),
    re.compile(r'\baccepted?\b', re.IGNORECASE),
    re.compile(r'\bconfirm\s+agent\s+session\b', re.IGNORECASE),
    re.compile(r'\bconfirm\s+session\b', re.IGNORECASE),
    re.compile(r'\bagent\s+session\b', re.IGNORECASE),
    re.compile(r'^\s*(?:yes|yeah|yep|ok(?:ay)?|sure|go\s+ahead|proceed|start\s+it|do\s+it)\s*$', re.IGNORECASE),
    re.compile(r'^\s*done\s+merge\s*$', re.IGNORECASE),
    re.compile(r'^\s*merge\s+done\s*$', re.IGNORECASE),
]

# Maximum word count for a message to be considered a "bare" confirmation.
# Longer messages are unlikely to be pure acknowledgements.
_MAX_WORDS_FOR_BARE_CONFIRMATION = 12


def _normalise(text: str) -> str:
    """Strip punctuation and lowercase *text* for fuzzy comparison."""
    return re.sub(r'[^\w\s]', '', text).lower().strip()


def is_confirmation_phrase(text: str) -> bool:
    """Return ``True`` if *text* looks like a bare confirmation with no new task.

    Parameters
    ----------
    text:
        Raw user input.

    Returns
    -------
    bool
        ``True`` when the message is recognised as a standalone confirmation
        rather than a substantive new request.
    """
    word_count = len(text.split())
    if word_count > _MAX_WORDS_FOR_BARE_CONFIRMATION:
        return False
    normalised = _normalise(text)
    return any(p.search(normalised) for p in _CONFIRMATION_PATTERNS)


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Represents a single agent work session."""

    session_id: str
    description: str
    status: str  # 'in_progress' | 'completed' | 'failed'
    started_at: float = field(default_factory=time.monotonic)
    completed_at: Optional[float] = None

    def complete(self) -> None:
        self.status = 'completed'
        self.completed_at = time.monotonic()

    def fail(self) -> None:
        self.status = 'failed'
        self.completed_at = time.monotonic()

    @property
    def elapsed(self) -> float:
        """Seconds since the session started."""
        end = self.completed_at or time.monotonic()
        return end - self.started_at

    @property
    def seconds_since_completion(self) -> Optional[float]:
        """Seconds elapsed since the session completed, or ``None`` if still running."""
        if self.completed_at is None:
            return None
        return time.monotonic() - self.completed_at


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class ChatSessionManager:
    """Guard against redundant task creation caused by repeated confirmations.

    Parameters
    ----------
    dedup_window_seconds:
        Seconds after a session completes during which a bare confirmation
        message will be suppressed rather than treated as a new task trigger.
        Defaults to 120 s (2 minutes).
    max_history:
        Maximum number of completed sessions to keep in memory for auditability.
    """

    def __init__(
        self,
        dedup_window_seconds: int = 120,
        max_history: int = 50,
    ) -> None:
        self._dedup_window = dedup_window_seconds
        self._max_history = max_history
        self._sessions: Dict[str, Session] = {}
        self._history: List[Session] = []

    # ------------------------------------------------------------------
    # Public helpers (module-level functions delegated here for convenience)
    # ------------------------------------------------------------------

    @staticmethod
    def is_confirmation_phrase(text: str) -> bool:
        """Proxy for the module-level :func:`is_confirmation_phrase`."""
        return is_confirmation_phrase(text)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, description: str) -> str:
        """Create and register a new in-progress session.

        Parameters
        ----------
        description:
            Human-readable summary of what the session is doing (used for
            logging and auditability).

        Returns
        -------
        str
            Unique session ID (UUID4).
        """
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            description=description,
            status='in_progress',
        )
        self._sessions[session_id] = session
        logger.info('ChatSessionManager: session started – %s (%s)', session_id, description)
        return session_id

    def complete_session(self, session_id: str) -> bool:
        """Mark *session_id* as completed.

        Returns
        -------
        bool
            ``True`` if the session was found and updated; ``False`` otherwise.
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            logger.warning('ChatSessionManager: complete called for unknown session %s', session_id)
            return False
        session.complete()
        self._archive(session)
        logger.info('ChatSessionManager: session completed – %s (%.1f s)', session_id, session.elapsed)
        return True

    def fail_session(self, session_id: str) -> bool:
        """Mark *session_id* as failed.

        Returns
        -------
        bool
            ``True`` if the session was found and updated; ``False`` otherwise.
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.fail()
        self._archive(session)
        logger.warning('ChatSessionManager: session failed – %s', session_id)
        return True

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def should_suppress(self, user_text: str) -> bool:
        """Return ``True`` if *user_text* should be suppressed as a duplicate.

        Suppression occurs when:

        * *user_text* is a bare confirmation phrase (detected by
          :func:`is_confirmation_phrase`), **and**
        * At least one session is currently ``in_progress``, **or**
        * A session completed within the dedup window.

        Parameters
        ----------
        user_text:
            Raw user message.

        Returns
        -------
        bool
            ``True`` → suppress (don't start new work).
            ``False`` → allow (treat as genuine new request).
        """
        if not is_confirmation_phrase(user_text):
            return False

        # Any in-progress session → suppress.
        if self._sessions:
            logger.debug(
                'ChatSessionManager: suppressing confirmation – %d session(s) in progress',
                len(self._sessions),
            )
            return True

        # Recent completion → suppress.
        recent = self._last_completed_session()
        if recent is not None:
            age = recent.seconds_since_completion or 0.0
            if age <= self._dedup_window:
                logger.debug(
                    'ChatSessionManager: suppressing confirmation – last session completed %.0f s ago '
                    '(dedup window: %d s)',
                    age,
                    self._dedup_window,
                )
                return True

        return False

    def suppression_reason(self, user_text: str) -> str:
        """Return a human-readable explanation of why *user_text* is suppressed.

        Returns an empty string when suppression is not applicable.
        """
        if not is_confirmation_phrase(user_text):
            return ''

        if self._sessions:
            ids = list(self._sessions.keys())
            return (
                f"A session is already in progress ({ids[0][:8]}…). "
                "Your confirmation has been acknowledged – no new task was started."
            )

        recent = self._last_completed_session()
        if recent is not None:
            age = recent.seconds_since_completion or 0.0
            if age <= self._dedup_window:
                return (
                    f"The previous session '{recent.description[:60]}' completed "
                    f"{age:.0f} s ago. Your confirmation has been acknowledged – "
                    "no new task was started."
                )

        return ''

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def active_session_count(self) -> int:
        """Return the number of currently in-progress sessions."""
        return len(self._sessions)

    def session_history(self) -> List[Session]:
        """Return a copy of the completed/failed session archive."""
        return list(self._history)

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return the live session for *session_id*, or ``None``."""
        return self._sessions.get(session_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _archive(self, session: Session) -> None:
        self._history.append(session)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def _last_completed_session(self) -> Optional[Session]:
        """Return the most recently completed/failed session, or ``None``."""
        completed = [s for s in self._history if s.completed_at is not None]
        if not completed:
            return None
        return max(completed, key=lambda s: s.completed_at or 0.0)
