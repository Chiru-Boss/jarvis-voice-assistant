import requests

NVIDIA_API_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'

SYSTEM_PROMPT = (
    "You are JARVIS, a professional and highly capable virtual assistant. "
    "You provide accurate, helpful, and concise information. "
    "Keep responses brief (2-3 sentences) and suitable for voice output. "
    "Maintain a professional yet approachable tone at all times."
)


def process_input(user_input, conversation_history=None,
                  api_key='', model='meta/llama-3.1-8b-instruct',
                  temperature=0.7, max_tokens=300, timeout=60):
    """Send *user_input* to the NVIDIA Llama endpoint and return the response.

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

    Returns
    -------
    str
        The assistant's reply, or an error description on failure.
    """
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({'role': 'user', 'content': user_input})

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
    }
    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }

    try:
        response = requests.post(
            NVIDIA_API_URL, headers=headers, json=payload, timeout=timeout
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return f"API error {response.status_code}: {response.text[:200]}"
    except requests.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {e}"
