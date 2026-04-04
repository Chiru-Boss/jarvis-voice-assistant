import os
from dotenv import load_dotenv

load_dotenv()


def _safe_int(value, default=10):
    """Parse *value* as an integer, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"⚠️  Invalid integer config value '{value}', using default {default}.")
        return default


CONFIG = {
    'NVIDIA_API_KEY': os.getenv('NVIDIA_API_KEY', ''),
    'NVIDIA_LLM_MODEL': os.getenv('NVIDIA_LLM_MODEL', 'meta/llama-3.1-8b-instruct'),
    'NVIDIA_API_URL': 'https://integrate.api.nvidia.com/v1/chat/completions',

    'ELEVENLABS_API_KEY': os.getenv('ELEVENLABS_API_KEY', ''),
    'ELEVENLABS_VOICE_ID': os.getenv('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM'),
    'ELEVENLABS_MODEL': os.getenv('ELEVENLABS_MODEL', 'eleven_monolingual_v1'),

    'WAKE_WORD': os.getenv('WAKE_WORD', 'jarvis').lower(),
    'VOICE_ENABLED': os.getenv('VOICE_ENABLED', 'true').lower() == 'true',
    'MEMORY_FILE': os.getenv('MEMORY_FILE', 'jarvis_memory.json'),
    'MAX_HISTORY': _safe_int(os.getenv('MAX_HISTORY', '10'), default=10),

    # Audio settings
    'SAMPLE_RATE': 16000,
    'CHANNELS': 1,
    'CHUNK_SIZE': 320,       # 20ms frames at 16kHz (required by webrtcvad)
    'SILENCE_TIMEOUT': 1.5,  # seconds of silence before stopping
    'VAD_AGGRESSIVENESS': 2, # 0-3, higher = more aggressive filtering

    # LLM settings
    'TEMPERATURE': 0.7,
    'MAX_TOKENS': 300,
    'REQUEST_TIMEOUT': 60,
}
