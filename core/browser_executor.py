from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_BROWSER_URL = 'https://www.google.com'


class BrowserExecutor:
    """Playwright-based browser automation with selector strategy caching."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        use_playwright: bool = False,
        cache_path: str = 'data/browser_strategies.json',
    ) -> None:
        self.enabled = enabled
        self.use_playwright = use_playwright
        self.cache_path = Path(cache_path)
        self._strategy_cache = self._load_cache()

    def run_task(self, *, url: str, action: str, target: str = '', value: str = '') -> Dict[str, str]:
        if not self.enabled:
            return {'status': 'disabled', 'message': 'Browser automation disabled.'}
        if not self.use_playwright:
            return {'status': 'disabled', 'message': 'USE_PLAYWRIGHT is false.'}

        try:
            from playwright.sync_api import TimeoutError, sync_playwright  # type: ignore
        except ImportError:
            return {'status': 'error', 'message': 'Playwright is not installed.'}
        except Exception as exc:
            return {'status': 'error', 'message': f'Playwright import failed: {exc}'}

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False)
                page = browser.new_page()
                page.goto(url, wait_until='domcontentloaded')

                if action == 'scroll':
                    self._smart_scroll(page, target_text=target)
                elif action in {'click', 'type'}:
                    selector = self._resolve_selector(target)
                    locator = page.locator(selector).first
                    self._wait_for_element_stability(locator)
                    if action == 'click':
                        locator.click()
                    else:
                        locator.fill(value)
                    self._store_strategy(target, selector)
                elif action == 'navigate':
                    page.goto(value or url, wait_until='domcontentloaded')
                else:
                    return {'status': 'error', 'message': f'Unsupported action: {action}'}

                current_url = page.url
                browser.close()
                return {'status': 'ok', 'message': f'Action {action} completed.', 'url': current_url}
        except TimeoutError as exc:
            return {'status': 'error', 'message': f'Playwright timeout: {exc}'}
        except Exception as exc:
            return {'status': 'error', 'message': f'Playwright execution failed: {exc}'}

    def _resolve_selector(self, semantic_target: str) -> str:
        if semantic_target in self._strategy_cache:
            return self._strategy_cache[semantic_target]
        return self._candidate_selectors(semantic_target)[0]

    @staticmethod
    def _candidate_selectors(label: str) -> List[str]:
        clean = label.strip()
        escaped = clean.replace('"', '\\"')
        return [
            f'[data-testid="{escaped}"]',
            f'[aria-label="{escaped}"]',
            f'text="{escaped}"',
            f'button:has-text("{escaped}")',
            f'input[placeholder="{escaped}"]',
        ]

    @staticmethod
    def _wait_for_element_stability(locator: object) -> None:
        locator.wait_for(state='visible', timeout=5000)  # type: ignore[attr-defined]
        locator.wait_for(state='attached', timeout=5000)  # type: ignore[attr-defined]

    @staticmethod
    def _smart_scroll(page: object, *, target_text: str = '', max_steps: int = 15) -> None:
        if not target_text:
            for _ in range(max_steps):
                page.mouse.wheel(0, 700)  # type: ignore[attr-defined]
            return
        locator = page.locator(f'text="{target_text}"').first  # type: ignore[attr-defined]
        for _ in range(max_steps):
            if locator.is_visible():  # type: ignore[attr-defined]
                break
            page.mouse.wheel(0, 700)  # type: ignore[attr-defined]

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _store_strategy(self, semantic_target: str, selector: str) -> None:
        if not semantic_target:
            return
        self._strategy_cache[semantic_target] = selector
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._strategy_cache, indent=2), encoding='utf-8')
