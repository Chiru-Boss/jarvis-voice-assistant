"""System Health – verify that all JARVIS subsystems are operational.

Call :func:`check_health` to get a :class:`HealthReport` that lists every
subsystem and whether it loaded successfully.  The report is intentionally
dependency-light: it only attempts ``import`` of each module and a basic
instantiation check; it never opens hardware devices or network connections.

Typical usage
-------------
>>> from core.system_health import check_health
>>> report = check_health()
>>> print(report.summary())
>>> assert report.healthy
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Subsystem registry
# Each entry is (display_name, dotted_module_path).
# Modules that require optional hardware (MediaPipe, OpenCV …) are listed
# under "optional" and only degrade the status to "partial" rather than
# "unhealthy" when absent.
# ---------------------------------------------------------------------------

_REQUIRED_SUBSYSTEMS: List[tuple[str, str]] = [
    # Voice pipeline (logic modules only – hardware probed separately)
    ("Speech Recognition",   "core.speech_recognition"),
    ("Wake Word",            "core.wake_word"),
    ("LLM Brain",            "core.llm_brain"),
    # MCP tool architecture
    ("Tool Registry",        "core.tool_registry"),
    ("MCP Server",           "core.mcp_server"),
    ("MCP Client",           "core.mcp_client"),
    # Adaptive agent subsystems
    ("Pattern Memory",       "core.pattern_memory"),
    ("Behavior Learner",     "core.behavior_learner"),
    ("Prediction Engine",    "core.prediction_engine"),
    ("Adaptive Agent",       "core.adaptive_agent"),
    ("App Controller",       "core.app_controller"),
    ("System Executor",      "core.system_executor"),
    ("Browser Automation",   "core.browser_automation"),
    ("Screen Vision",        "core.screen_vision"),
    # Tools
    ("System Tools",         "tools.system_tools"),
    ("Laptop Control",       "tools.laptop_control"),
    ("Knowledge Base",       "tools.knowledge_base"),
    ("Home Automation",      "tools.home_automation"),
    ("Web APIs",             "tools.web_apis"),
]

_OPTIONAL_SUBSYSTEMS: List[tuple[str, str]] = [
    # Hardware-dependent voice I/O – requires PortAudio / audio device
    ("Audio Input",           "core.audio_input"),
    ("Text-to-Speech",        "core.text_to_speech"),
    # Hand-tracking stack requires mediapipe / opencv which are optional
    ("Swipe Keyboard",       "core.swipe_keyboard"),
    ("Hand Tracking",        "core.hand_tracking"),
    ("Gesture Recognition",  "core.gesture_recognition"),
    ("Hand Mouse Controller","core.hand_mouse_controller"),
    ("Hand UI Overlay",      "core.hand_ui_overlay"),
    ("Hand-Voice Integration","core.hand_voice_integration"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubsystemStatus:
    """Status of a single subsystem."""
    name: str
    ok: bool
    optional: bool = False
    error: str = ""

    def __str__(self) -> str:
        status = "✅ OK" if self.ok else ("⚠️  MISSING" if self.optional else "❌ FAIL")
        suffix = f" – {self.error}" if self.error else ""
        return f"  {status}  {self.name}{suffix}"


@dataclass
class HealthReport:
    """Aggregated health report for all JARVIS subsystems."""
    statuses: List[SubsystemStatus] = field(default_factory=list)

    # ── Derived properties ──────────────────────────────────────────────

    @property
    def required_ok(self) -> bool:
        """True when every *required* subsystem loaded without error."""
        return all(s.ok for s in self.statuses if not s.optional)

    @property
    def optional_ok(self) -> bool:
        """True when every *optional* subsystem is also available."""
        return all(s.ok for s in self.statuses if s.optional)

    @property
    def healthy(self) -> bool:
        """True when all required subsystems are operational."""
        return self.required_ok

    @property
    def status_label(self) -> str:
        """Human-readable overall status label."""
        if self.required_ok and self.optional_ok:
            return "FULLY OPERATIONAL"
        if self.required_ok:
            return "OPERATIONAL (optional modules absent)"
        return "DEGRADED"

    # ── Summary text ────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a multi-line summary string suitable for console output."""
        lines: List[str] = [
            "╔══════════════════════════════════════════════════════╗",
            "║           JARVIS – System Health Report              ║",
            "╠══════════════════════════════════════════════════════╣",
        ]

        required = [s for s in self.statuses if not s.optional]
        optional = [s for s in self.statuses if s.optional]

        lines.append("║ Required subsystems:                                 ║")
        for s in required:
            lines.append(str(s))

        if optional:
            lines.append("║ Optional subsystems:                                 ║")
            for s in optional:
                lines.append(str(s))

        req_pass  = sum(1 for s in required if s.ok)
        opt_pass  = sum(1 for s in optional if s.ok)
        lines += [
            "╠══════════════════════════════════════════════════════╣",
            f"  Required: {req_pass}/{len(required)} OK",
            f"  Optional: {opt_pass}/{len(optional)} OK",
            f"  Overall status: {self.status_label}",
            "╚══════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    # ── Dict / JSON-friendly representation ─────────────────────────────

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable dict of the report."""
        return {
            "status":   self.status_label,
            "healthy":  self.healthy,
            "required": {s.name: {"ok": s.ok, "error": s.error}
                         for s in self.statuses if not s.optional},
            "optional": {s.name: {"ok": s.ok, "error": s.error}
                         for s in self.statuses if s.optional},
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_health() -> HealthReport:
    """Import every registered subsystem and return a :class:`HealthReport`.

    The check is purely static (no hardware, no network).  Already-imported
    modules are reused from ``sys.modules`` to keep the call fast.
    """
    report = HealthReport()

    for name, module_path in _REQUIRED_SUBSYSTEMS:
        _probe(report, name, module_path, optional=False)

    for name, module_path in _OPTIONAL_SUBSYSTEMS:
        _probe(report, name, module_path, optional=True)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe(report: HealthReport, name: str, module_path: str,
           *, optional: bool) -> None:
    """Try to import *module_path* and append a :class:`SubsystemStatus`."""
    try:
        if module_path not in sys.modules:
            importlib.import_module(module_path)
        report.statuses.append(SubsystemStatus(name=name, ok=True,
                                                optional=optional))
    except Exception as exc:  # pragma: no cover – hardware-dependent paths
        report.statuses.append(SubsystemStatus(
            name=name, ok=False, optional=optional,
            error=f"{type(exc).__name__}: {exc}",
        ))
