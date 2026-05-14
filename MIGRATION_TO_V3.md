# Migration to JARVIS v3

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Configure `.env`

Copy `.env.example` to `.env` and keep all v3 flags disabled initially.

## 3) Enable features gradually

```env
VISION_ENABLED=true
BROWSER_AUTOMATION_ENABLED=true
PTT_ENABLED=true
ENABLE_OVERLAY_UI=true
PERSONA_SYSTEM_ENABLED=true
SELF_HEALING_ENABLED=true
```

## 4) Verify

```bash
python -m pytest tests/ -v
```

v3 is additive and does not replace existing hand-tracking, adaptive agent, or MCP workflows.
