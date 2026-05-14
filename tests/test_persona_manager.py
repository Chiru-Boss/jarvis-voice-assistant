"""Tests for core/persona_manager.py."""

from __future__ import annotations

import tempfile
import unittest

from core.persona_manager import PersonaManager


class TestPersonaManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = PersonaManager(data_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_and_load_persona(self):
        self.manager.save_persona('research_specialist', 'Find and compare sources')
        persona = self.manager.load_persona('research_specialist')
        self.assertEqual(persona['name'], 'research_specialist')

    def test_switch_persona_sets_active_name(self):
        self.manager.save_persona('shopping_buddy', 'Find cheapest options')
        self.manager.switch_persona('shopping_buddy')
        self.assertEqual(self.manager.active_persona, 'shopping_buddy')


if __name__ == '__main__':
    unittest.main()
