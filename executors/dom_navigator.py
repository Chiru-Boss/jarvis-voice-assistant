"""Semantic DOM targeting helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DOMNavigator:
    """Score and select the best interactive element for a user goal."""

    _INTERACTIVE_TAGS = {'button', 'a', 'input', 'textarea', 'select'}

    def rank_elements(self, elements: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        query = (query or '').lower()
        ranked = []
        for el in elements:
            score = 0.0
            tag = str(el.get('tag', '')).lower()
            text = str(el.get('text', '')).lower()
            visible = bool(el.get('visible', True))
            if visible:
                score += 2.0
            if tag in self._INTERACTIVE_TAGS:
                score += 2.5
            if query and query in text:
                score += 3.0
            score += float(el.get('importance', 0.0))
            ranked.append({**el, '_score': score})

        ranked.sort(key=lambda item: item.get('_score', 0.0), reverse=True)
        return ranked

    def best_match(self, elements: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
        ranked = self.rank_elements(elements, query)
        return ranked[0] if ranked else None
