# Persona System

`core/persona_manager.py` stores persona profiles in JSON files.

## Environment

```env
PERSONA_SYSTEM_ENABLED=true
PERSONA_DATA_DIR=data/personas
```

## Example

```python
from core.persona_manager import PersonaManager

pm = PersonaManager('data/personas')
pm.save_persona('research_specialist', 'Be citation-heavy and precise')
pm.switch_persona('research_specialist')
```
