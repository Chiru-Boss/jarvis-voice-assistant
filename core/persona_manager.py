"""Persona profile save/load/switch support for multi-expert agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class PersonaManager:
    def __init__(self, data_dir: str = 'data/personas'):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._active_persona: Optional[str] = None

    def save_persona(self, name: str, instructions: str, metadata: Optional[Dict[str, str]] = None) -> Path:
        payload = {
            'name': name,
            'instructions': instructions,
            'metadata': metadata or {},
        }
        path = self._dir / f'{name}.json'
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return path

    def load_persona(self, name: str) -> Dict[str, object]:
        path = self._dir / f'{name}.json'
        return json.loads(path.read_text(encoding='utf-8'))

    def list_personas(self) -> List[str]:
        return sorted(path.stem for path in self._dir.glob('*.json'))

    def switch_persona(self, name: str) -> Dict[str, object]:
        persona = self.load_persona(name)
        self._active_persona = name
        return persona

    @property
    def active_persona(self) -> Optional[str]:
        return self._active_persona
