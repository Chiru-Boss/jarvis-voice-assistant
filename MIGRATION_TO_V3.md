# Migration to JARVIS v3

JARVIS v3 adds a Whisperflowactions-style orchestration layer while keeping all
existing systems available.

## 1) Backward compatibility guarantees

- Hand tracking and air-swipe keyboard are unchanged.
- Pattern learning (`pattern_memory`, `behavior_learner`, `prediction_engine`) is unchanged.
- MCP tool architecture remains intact.
- Existing voice commands (`patterns`, `predict`, `memory`, etc.) remain available.
- New v3 features are disabled by default.

## 2) Enable v3 features gradually

Set these in `.env`:

```env
VISION_MODEL=llama-3.2-90b-vision-instruct
VISION_ENABLED=true
BROWSER_AUTOMATION_ENABLED=true
USE_PLAYWRIGHT=true
PTT_ENABLED=true
ENABLE_OVERLAY_UI=true
PERSONA_SYSTEM_ENABLED=true
CONFIRMATION_MODE=true
```

Optional local model routing:

```env
PRIMARY_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

## 3) New core entrypoint

- New orchestrator: `core/orchestrator.py` (`JarvisOrchestrator`)
- Existing `main.py` is still valid and unchanged for safe rollout.

## 4) Browser automation setup

Install Playwright browser binaries:

```bash
playwright install chromium
```

## 5) Test coverage added for v3

- `tests/test_vision_engine.py`
- `tests/test_browser_executor.py`
- `tests/test_self_healer.py`
