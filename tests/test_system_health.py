"""Tests for core/system_health.py – JARVIS system health checker.

Covers:
- SubsystemStatus string representation
- HealthReport derived properties (healthy, status_label, required_ok, optional_ok)
- HealthReport.summary() and as_dict() output shapes
- check_health() returns a HealthReport with all expected sections
- All required subsystems report OK in this environment
"""

from __future__ import annotations

import os
import sys
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.system_health import (
    HealthReport,
    SubsystemStatus,
    check_health,
    _REQUIRED_SUBSYSTEMS,
    _OPTIONAL_SUBSYSTEMS,
)


class TestSubsystemStatus(unittest.TestCase):
    """Unit tests for SubsystemStatus string representation."""

    def test_ok_status_string(self):
        s = SubsystemStatus(name="Foo", ok=True)
        self.assertIn("OK", str(s))
        self.assertIn("Foo", str(s))

    def test_fail_status_string(self):
        s = SubsystemStatus(name="Bar", ok=False, optional=False,
                            error="ImportError: no module")
        text = str(s)
        self.assertIn("FAIL", text)
        self.assertIn("Bar", text)
        self.assertIn("ImportError", text)

    def test_optional_missing_status_string(self):
        s = SubsystemStatus(name="Baz", ok=False, optional=True)
        self.assertIn("MISSING", str(s))

    def test_no_error_suffix_when_ok(self):
        s = SubsystemStatus(name="X", ok=True)
        self.assertNotIn("–", str(s))


class TestHealthReportProperties(unittest.TestCase):
    """Unit tests for HealthReport derived properties."""

    def _report(self, required_ok: bool, optional_ok: bool) -> HealthReport:
        r = HealthReport()
        r.statuses.append(SubsystemStatus("Req", ok=required_ok, optional=False))
        r.statuses.append(SubsystemStatus("Opt", ok=optional_ok, optional=True))
        return r

    def test_healthy_when_required_ok(self):
        self.assertTrue(self._report(True, False).healthy)

    def test_unhealthy_when_required_fails(self):
        self.assertFalse(self._report(False, True).healthy)

    def test_required_ok_property(self):
        self.assertTrue(self._report(True, False).required_ok)
        self.assertFalse(self._report(False, True).required_ok)

    def test_optional_ok_property(self):
        self.assertTrue(self._report(True, True).optional_ok)
        self.assertFalse(self._report(True, False).optional_ok)

    def test_status_label_fully_operational(self):
        self.assertEqual(self._report(True, True).status_label, "FULLY OPERATIONAL")

    def test_status_label_operational_optional_absent(self):
        label = self._report(True, False).status_label
        self.assertIn("OPERATIONAL", label)
        self.assertIn("optional", label)

    def test_status_label_degraded(self):
        self.assertEqual(self._report(False, False).status_label, "DEGRADED")


class TestHealthReportSummary(unittest.TestCase):
    """Unit tests for HealthReport.summary() output."""

    def setUp(self):
        r = HealthReport()
        r.statuses.append(SubsystemStatus("Engine", ok=True, optional=False))
        r.statuses.append(SubsystemStatus("Camera", ok=False, optional=True,
                                           error="No device"))
        self.report = r

    def test_summary_contains_header(self):
        self.assertIn("JARVIS", self.summary)

    def test_summary_contains_required_section(self):
        self.assertIn("Required", self.summary)

    def test_summary_contains_optional_section(self):
        self.assertIn("Optional", self.summary)

    def test_summary_contains_status_label(self):
        self.assertIn(self.report.status_label, self.summary)

    @property
    def summary(self) -> str:
        return self.report.summary()


class TestHealthReportAsDict(unittest.TestCase):
    """Unit tests for HealthReport.as_dict() shape."""

    def setUp(self):
        r = HealthReport()
        r.statuses.append(SubsystemStatus("A", ok=True,  optional=False))
        r.statuses.append(SubsystemStatus("B", ok=False, optional=True, error="x"))
        self.d = r.as_dict()

    def test_top_level_keys(self):
        self.assertIn("status",   self.d)
        self.assertIn("healthy",  self.d)
        self.assertIn("required", self.d)
        self.assertIn("optional", self.d)

    def test_required_entry_shape(self):
        entry = self.d["required"]["A"]
        self.assertIn("ok",    entry)
        self.assertIn("error", entry)

    def test_optional_entry_shape(self):
        entry = self.d["optional"]["B"]
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["error"], "x")

    def test_healthy_matches_report(self):
        r = HealthReport()
        r.statuses.append(SubsystemStatus("X", ok=True, optional=False))
        self.assertTrue(r.as_dict()["healthy"])


class TestCheckHealth(unittest.TestCase):
    """Integration tests for check_health()."""

    @classmethod
    def setUpClass(cls):
        cls.report = check_health()

    def test_returns_health_report(self):
        self.assertIsInstance(self.report, HealthReport)

    def test_has_statuses(self):
        self.assertTrue(len(self.report.statuses) > 0)

    def test_required_subsystems_all_present(self):
        """Every registered required subsystem must appear in the report."""
        names_in_report = {s.name for s in self.report.statuses}
        for display_name, _ in _REQUIRED_SUBSYSTEMS:
            self.assertIn(display_name, names_in_report,
                          f"Required subsystem '{display_name}' missing from report")

    def test_optional_subsystems_all_present(self):
        """Every registered optional subsystem must appear in the report."""
        names_in_report = {s.name for s in self.report.statuses}
        for display_name, _ in _OPTIONAL_SUBSYSTEMS:
            self.assertIn(display_name, names_in_report,
                          f"Optional subsystem '{display_name}' missing from report")

    def test_required_subsystems_ok(self):
        """All required subsystems must be importable in the test environment."""
        failed = [s for s in self.report.statuses if not s.optional and not s.ok]
        self.assertEqual(failed, [],
                         msg="\n".join(f"  {s}" for s in failed))

    def test_healthy(self):
        self.assertTrue(self.report.healthy,
                        msg=self.report.summary())

    def test_summary_is_string(self):
        self.assertIsInstance(self.report.summary(), str)

    def test_as_dict_is_dict(self):
        self.assertIsInstance(self.report.as_dict(), dict)

    def test_as_dict_healthy_true(self):
        d = self.report.as_dict()
        self.assertTrue(d["healthy"])

    def test_status_label_not_degraded(self):
        self.assertNotEqual(self.report.status_label, "DEGRADED")


if __name__ == "__main__":
    unittest.main()
