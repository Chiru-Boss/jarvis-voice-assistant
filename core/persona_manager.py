from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PersonaManager:
    """Persist and manage multi-persona agent profiles."""

    def __init__(self, path: str = 'data/personas.json') -> None:
        self.path = Path(path)
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {'active': None, 'personas': {}}
        try:
            return json.loads(self.path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug('Failed to load persona store: %s', exc)
            return {'active': None, 'personas': {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding='utf-8')

    def save_persona(self, name: str, *, system_prompt: str, model: str = '', temperature: float = 0.7) -> Dict[str, Any]:
        personas = self._data.setdefault('personas', {})
        personas[name] = {
            'name': name,
            'system_prompt': system_prompt,
            'model': model,
            'temperature': temperature,
        }
        if not self._data.get('active'):
            self._data['active'] = name
        self._save()
        return personas[name]  # type: ignore[index]

    def load_persona(self, name: str) -> Optional[Dict[str, Any]]:
        personas = self._data.get('personas', {})
        return personas.get(name) if isinstance(personas, dict) else None

    def list_personas(self) -> List[str]:
        personas = self._data.get('personas', {})
        if not isinstance(personas, dict):
            return []
        return sorted(personas.keys())

    def set_active_persona(self, name: str) -> bool:
        if not self.load_persona(name):
            return False
        self._data['active'] = name
        self._save()
        return True

    def get_active_persona(self) -> Optional[Dict[str, Any]]:
        active_name = self._data.get('active')
        if not active_name:
            return None
        return self.load_persona(str(active_name))
