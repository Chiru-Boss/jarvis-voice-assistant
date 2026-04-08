"""Web API Tools – weather, web search, news headlines, crypto prices."""

from __future__ import annotations

import os
import urllib.parse
from typing import Any, Dict, List

import requests

from core.tool_registry import ToolRegistry

_REQUEST_TIMEOUT = 10
_USER_AGENT = 'JARVIS/2.0 (+https://github.com/Chiru-Boss/jarvis-voice-assistant)'


# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------

def get_weather(location: str) -> Dict[str, Any]:
    """Get current weather for *location* using the OpenWeatherMap free tier.

    Requires the ``OPENWEATHER_API_KEY`` environment variable.
    """
    api_key = os.getenv('OPENWEATHER_API_KEY', '')
    if not api_key:
        return {
            'error': (
                'OPENWEATHER_API_KEY is not set. '
                'Get a free key at https://openweathermap.org/api'
            )
        }

    try:
        url = (
            'https://api.openweathermap.org/data/2.5/weather'
            f'?q={urllib.parse.quote(location)}'
            f'&appid={api_key}&units=metric'
        )
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={'User-Agent': _USER_AGENT})
        if resp.status_code == 200:
            data = resp.json()
            return {
                'location': data.get('name', location),
                'temperature_c': data['main']['temp'],
                'feels_like_c': data['main']['feels_like'],
                'description': data['weather'][0]['description'],
                'humidity_percent': data['main']['humidity'],
                'wind_speed_ms': data['wind']['speed'],
            }
        return {'error': f"OpenWeatherMap API error {resp.status_code}: {resp.text[:200]}"}
    except Exception as exc:
        return {'error': str(exc)}


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search the web via DuckDuckGo Instant Answer API (free, no key needed)."""
    try:
        url = (
            'https://api.duckduckgo.com/'
            f'?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1'
        )
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={'User-Agent': _USER_AGENT})
        data = resp.json()
        results: List[Dict[str, Any]] = []

        # Instant answer / abstract
        if data.get('AbstractText'):
            results.append({
                'title': data.get('Heading', query),
                'snippet': data['AbstractText'],
                'url': data.get('AbstractURL', ''),
            })

        # Related topics
        for topic in data.get('RelatedTopics', []):
            if isinstance(topic, dict) and topic.get('Text'):
                results.append({
                    'title': topic['Text'][:80],
                    'snippet': topic['Text'],
                    'url': topic.get('FirstURL', ''),
                })
            if len(results) >= max_results:
                break

        return results if results else [{'snippet': 'No results found for the query.'}]

    except Exception as exc:
        return [{'error': str(exc)}]


def get_news(topic: str = 'technology', max_results: int = 5) -> List[Dict[str, Any]]:
    """Get latest news headlines on *topic* via DuckDuckGo (free, no key needed)."""
    try:
        url = (
            'https://api.duckduckgo.com/'
            f'?q={urllib.parse.quote(topic + " news")}&format=json&no_html=1'
        )
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={'User-Agent': _USER_AGENT})
        data = resp.json()
        results: List[Dict[str, Any]] = []

        for item in data.get('RelatedTopics', []):
            if isinstance(item, dict) and item.get('Text'):
                results.append({
                    'headline': item['Text'][:120],
                    'url': item.get('FirstURL', ''),
                })
            if len(results) >= max_results:
                break

        return results if results else [{'headline': f'No news found for "{topic}".'}]

    except Exception as exc:
        return [{'error': str(exc)}]


def get_crypto_prices(coins: str = 'bitcoin,ethereum') -> Dict[str, Any]:
    """Get current cryptocurrency prices (USD) via CoinGecko (free, no key needed)."""
    try:
        ids = urllib.parse.quote(coins.lower().strip())
        url = (
            'https://api.coingecko.com/api/v3/simple/price'
            f'?ids={ids}&vs_currencies=usd&include_24hr_change=true'
        )
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={'User-Agent': _USER_AGENT})
        if resp.status_code == 200:
            return resp.json()
        return {'error': f"CoinGecko API error {resp.status_code}: {resp.text[:200]}"}
    except Exception as exc:
        return {'error': str(exc)}


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def register_tools(registry: ToolRegistry) -> None:
    """Register all web API tools with *registry*."""

    registry.register(
        name='get_weather',
        description=(
            'Get the current weather for a city or location. '
            'Returns temperature, description, humidity, and wind speed. '
            'Requires OPENWEATHER_API_KEY environment variable (free tier).'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'City name, e.g. "London" or "New York, US".',
                },
            },
            'required': ['location'],
        },
        func=get_weather,
        safe=True,
    )

    registry.register(
        name='web_search',
        description=(
            'Search the web using DuckDuckGo and return top results with snippets. '
            'No API key required.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Search query string.',
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of results to return (default 5).',
                },
            },
            'required': ['query'],
        },
        func=web_search,
        safe=True,
    )

    registry.register(
        name='get_news',
        description=(
            'Get the latest news headlines on a given topic using DuckDuckGo. '
            'No API key required.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'topic': {
                    'type': 'string',
                    'description': 'News topic, e.g. "technology", "sports", "AI".',
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of headlines to return (default 5).',
                },
            },
        },
        func=get_news,
        safe=True,
    )

    registry.register(
        name='get_crypto_prices',
        description=(
            'Get current cryptocurrency prices in USD with 24-hour change percentage. '
            'Uses CoinGecko (100% free, no API key needed).'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'coins': {
                    'type': 'string',
                    'description': (
                        'Comma-separated CoinGecko coin IDs, '
                        'e.g. "bitcoin,ethereum,dogecoin".'
                    ),
                },
            },
        },
        func=get_crypto_prices,
        safe=True,
    )
