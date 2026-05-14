# Browser Automation (Playwright)

## Environment

```env
BROWSER_AUTOMATION_ENABLED=true
USE_PLAYWRIGHT=true
BROWSER_HEADLESS=false
BROWSER_TIMEOUT=30
```

## Modules

- `core/browser_executor.py`: high-level automation wrapper
- `executors/browser_executor.py`: optional Playwright executor
- `executors/dom_navigator.py`: semantic element ranking and below-the-fold targeting support helpers

If Playwright is unavailable, execution fails safely with a clear message.
