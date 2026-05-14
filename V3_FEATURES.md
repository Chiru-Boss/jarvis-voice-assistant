# JARVIS v3 Features

- NVIDIA vision verification (`core/vision_engine.py`)
- Playwright-ready browser automation (`core/browser_executor.py`, `executors/`)
- Self-healing retries and diagnosis (`core/self_healer.py`)
- Intent routing (`core/action_router.py`, `core/intent_classifier.py`)
- Multi-expert personas (`core/persona_manager.py`)
- Push-to-talk controller (`core/ptt_controller.py`)
- HUD/tray state stack (`ui/overlay.py`, `ui/hud_state_manager.py`, `ui/tray.py`)
- Unified optional orchestrator (`core/orchestrator.py`)

All features are optional and disabled by default.
