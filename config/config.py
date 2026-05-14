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


def _safe_bool(value, default=False):
    """Parse *value* as boolean; accepts str/bool/int/None, case-insensitive."""
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {'1', 'true', 'yes', 'on'}:
        return True
    if lowered in {'0', 'false', 'no', 'off'}:
        return False
    print(f"⚠️  Invalid boolean config value '{value}', using default {default}.")
    return default


def _safe_float(value, default=0.0):
    """Parse *value* as a float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"⚠️  Invalid float config value '{value}', using default {default}.")
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

    # Experimental UI bridge (Whisperflowactions-style HUD integration prep)
    'UI_BRIDGE_ENABLED': _safe_bool(os.getenv('UI_BRIDGE_ENABLED', 'false'), default=False),
    'UI_BRIDGE_MAX_EVENTS': max(1, _safe_int(os.getenv('UI_BRIDGE_MAX_EVENTS', '200'), default=200)),

    # JARVIS v3 - Vision
    'VISION_ENABLED': _safe_bool(os.getenv('VISION_ENABLED', 'false'), default=False),
    'VISION_MODEL': os.getenv('VISION_MODEL', 'llama-3.2-90b-vision-instruct'),
    'NVIDIA_VISION_API_KEY': os.getenv('NVIDIA_VISION_API_KEY', ''),

    # JARVIS v3 - Browser automation
    'BROWSER_AUTOMATION_ENABLED': _safe_bool(os.getenv('BROWSER_AUTOMATION_ENABLED', 'false'), default=False),
    'USE_PLAYWRIGHT': _safe_bool(os.getenv('USE_PLAYWRIGHT', 'true'), default=True),
    'BROWSER_HEADLESS': _safe_bool(os.getenv('BROWSER_HEADLESS', 'false'), default=False),
    'BROWSER_TIMEOUT': max(5, _safe_int(os.getenv('BROWSER_TIMEOUT', '30'), default=30)),

    # JARVIS v3 - Push-to-talk
    'PTT_ENABLED': _safe_bool(os.getenv('PTT_ENABLED', 'false'), default=False),
    'PTT_HOTKEY': os.getenv('PTT_HOTKEY', 'ctrl+space'),

    # JARVIS v3 - Overlay
    'ENABLE_OVERLAY_UI': _safe_bool(os.getenv('ENABLE_OVERLAY_UI', 'false'), default=False),
    'OVERLAY_POSITION': os.getenv('OVERLAY_POSITION', 'top-right'),
    'OVERLAY_OPACITY': _safe_float(os.getenv('OVERLAY_OPACITY', '0.9'), default=0.9),

    # JARVIS v3 - Persona and routing
    'PERSONA_SYSTEM_ENABLED': _safe_bool(os.getenv('PERSONA_SYSTEM_ENABLED', 'false'), default=False),
    'PERSONA_DATA_DIR': os.getenv('PERSONA_DATA_DIR', 'data/personas'),
    'LLM_PROVIDER': os.getenv('LLM_PROVIDER', 'nvidia'),
    'FALLBACK_PROVIDERS': os.getenv('FALLBACK_PROVIDERS', 'ollama'),

    # JARVIS v3 - Self-healing
    'SELF_HEALING_ENABLED': _safe_bool(os.getenv('SELF_HEALING_ENABLED', 'false'), default=False),
    'MAX_RECOVERY_ATTEMPTS': max(1, _safe_int(os.getenv('MAX_RECOVERY_ATTEMPTS', '3'), default=3)),
}
