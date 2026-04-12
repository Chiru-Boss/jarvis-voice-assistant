from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests

from utils.helpers import truncate

logger = logging.getLogger(__name__)

NVIDIA_API_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'

SYSTEM_PROMPT = (
    "You are JARVIS, a professional and highly capable AI assistant with access to "
    "powerful tools. When a user asks something that requires real-time data, system "
    "access, file operations, or knowledge retrieval, use the appropriate tool. "
    "For conversational questions or general knowledge, respond directly without tools. "
    "Keep responses concise (2-4 sentences) and suitable for voice output. "
    "Always be professional, accurate, and helpful."
)


def _call_api(
    messages: List[Dict[str, Any]],
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Make a single call to the NVIDIA LLM API and return the parsed JSON."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
    }
    payload: Dict[str, Any] = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    if tools:
        payload['tools'] = tools
        payload['tool_choice'] = 'auto'

    response = requests.post(
        NVIDIA_API_URL, headers=headers, json=payload, timeout=timeout
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"API error {response.status_code}: {truncate(response.text, 200)}"
        )
    return response.json()


def process_input(
    user_input: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    api_key: str = '',
    model: str = 'meta/llama-3.1-8b-instruct',
    temperature: float = 0.7,
    max_tokens: int = 300,
    timeout: int = 60,
    mcp_client=None,
    max_tool_iterations: int = 5,
    pattern_hint: str = '',
) -> str:
    """Send *user_input* to the NVIDIA Llama endpoint and return the response.

    When *mcp_client* is provided the LLM may call any registered tool via
    OpenAI function-calling.  Tool calls are executed in a loop until the
    model produces a final text response or *max_tool_iterations* is reached.

    Parameters
    ----------
    user_input : str
        The latest message from the user.
    conversation_history : list[dict] or None
        OpenAI-style message list from recent exchanges (for context).
    api_key : str
        NVIDIA API key.
    model : str
        NVIDIA model identifier.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens in the response.
    timeout : int
        HTTP request timeout in seconds.
    mcp_client : MCPClient or None
        If provided, tool calling is enabled.  The client is used to
        discover available tools and execute tool calls returned by the LLM.
    max_tool_iterations : int
        Maximum number of tool-calling rounds before returning a fallback
        message (prevents infinite loops).
    pattern_hint : str
        Optional string summarising learned user patterns to include in the
        system prompt (e.g. "Most-used apps: brave, vscode; Common searches: python").

    Returns
    -------
    str
        The assistant's final reply, or an error description on failure.
    """
    # Build system prompt, optionally enriched with pattern context.
    system_content = SYSTEM_PROMPT
    if pattern_hint:
        system_content = (
            SYSTEM_PROMPT
            + f"\n\nUser behaviour context (use to personalise responses): {pattern_hint}"
        )

    messages: List[Dict[str, Any]] = [{'role': 'system', 'content': system_content}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({'role': 'user', 'content': user_input})

    tools = mcp_client.get_available_tools() if mcp_client else None

    for iteration in range(max_tool_iterations):
        try:
            response_data = _call_api(
                messages=messages,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                tools=tools,
            )
        except requests.Timeout:
            return 'Request timed out. Please try again.'
        except Exception as exc:
            return f'Error: {exc}'

        choice = response_data.get('choices', [{}])[0]
        message = choice.get('message', {})
        finish_reason = choice.get('finish_reason', '')
        tool_calls = message.get('tool_calls')

        # No tool calls – the model has produced a final text answer.
        if not tool_calls or finish_reason == 'stop':
            return message.get('content') or 'I was unable to generate a response.'

        # Append the assistant's message (with tool_calls) to the thread.
        messages.append(message)

        # Execute each tool call and append the results.
        for tc in tool_calls:
            tool_name = tc['function']['name']
            try:
                arguments = json.loads(tc['function'].get('arguments', '{}'))
            except json.JSONDecodeError:
                arguments = {}

            logger.info("Tool call: %s(%s)", tool_name, arguments)
            print(f'🔧 Tool call: {tool_name}({arguments})')
            tool_result = mcp_client.call_tool(tool_name, arguments)
            logger.info("Tool result: %s", truncate(str(tool_result), 200))
            print(f'✅ Tool result: {truncate(str(tool_result), 200)}')

            messages.append({
                'role': 'tool',
                'tool_call_id': tc.get('id', tool_name),
                'content': tool_result,
            })

        logger.debug("Tool iteration %d/%d completed", iteration + 1, max_tool_iterations)

    # Reached max iterations without a final text response.
    return (
        'I have gathered the necessary information. '
        'Please ask your question again for a summary.'
    )
