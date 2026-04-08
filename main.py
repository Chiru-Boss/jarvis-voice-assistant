"""JARVIS 2.0 – Professional AI Voice Assistant with MCP Tool Architecture.

Start the assistant:
    python main.py

Wake word: say "Jarvis" to activate, then speak your question.
The assistant stops listening automatically when you finish speaking.

Voice commands (after wake word):
  exit / quit / bye / shutdown  – shut down JARVIS
  memory / history / status     – display memory summary
  clear / forget / reset        – erase conversation memory
  tools                         – list available MCP tools
"""

import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import CONFIG
from config.tools_config import TOOLS_CONFIG
from core.audio_input import AudioInput
from core.speech_recognition import recognize_speech
from core.wake_word import listen_for_wake_word, strip_wake_word
from core.llm_brain import process_input
from core.text_to_speech import speak
from core.mcp_server import MCPServer
from core.mcp_client import MCPClient
from tools import build_registry
from utils.memory import ConversationMemory


# ---------------------------------------------------------------------------
# MCP initialisation
# ---------------------------------------------------------------------------

_TOOL_DESC_DISPLAY_LEN = 60  # Characters to show for each tool description in the banner


def init_mcp() -> MCPClient:
    """Build the tool registry, start the MCP server, and return a client."""
    registry = build_registry()
    server = MCPServer(
        registry=registry,
        approval_mode=TOOLS_CONFIG['APPROVAL_MODE'],
    )

    if TOOLS_CONFIG['MCP_SERVER_ENABLED']:
        server.start_http_server(
            host=TOOLS_CONFIG['MCP_SERVER_HOST'],
            port=TOOLS_CONFIG['MCP_SERVER_PORT'],
        )

    return MCPClient(server)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def startup_banner(memory: ConversationMemory, mcp_client: MCPClient) -> None:
    tool_count = len(mcp_client.get_available_tools())
    print('\n' + '=' * 70)
    print('         🤖 J.A.R.V.I.S 2.0 – PROFESSIONAL AI ASSISTANT 🤖')
    print('=' * 70)
    print('\n⚡ Initializing systems…')
    print('✅ NVIDIA Llama – Online')
    print('✅ Speech Recognition – Active')
    print('✅ Text-to-Speech – Ready')
    print(f'✅ Persistent Memory – {memory.summary()}')
    print(f'✅ MCP Tool Server – {tool_count} tools registered')
    if TOOLS_CONFIG['APPROVAL_MODE']:
        print('🔒 Approval Mode – ON (destructive tools require confirmation)')
    print(f'\n🔑 Wake word: "{CONFIG["WAKE_WORD"].capitalize()}"')
    print('💡 Commands after wake word: exit | memory | clear | tools')
    print('=' * 70 + '\n')


def handle_special_commands(text: str, memory: ConversationMemory, mcp_client: MCPClient) -> bool:
    """Check for built-in commands.  Return True if handled, False otherwise."""
    text_lower = text.lower().strip()

    if text_lower in ('exit', 'quit', 'bye', 'shutdown', 'close'):
        print('\n🛑 JARVIS shutting down. Goodbye! 👋\n')
        sys.exit(0)

    if text_lower in ('memory', 'history', 'status'):
        print(f'💾 Memory: {memory.summary()}\n')
        for i, conv in enumerate(memory.get_memory()[-5:], 1):
            ts = conv.get('timestamp')
            ts_display = ts[:19] if ts else 'N/A'
            print(f'  [{i}] {ts_display}')
            print(f'       You: {conv["user_input"][:60]}')
            print(f'       JARVIS: {conv["bot_response"][:60]}')
        print()
        return True

    if text_lower in ('clear', 'forget', 'reset'):
        memory.clear_memory()
        print('💾 Memory cleared!\n')
        return True

    if text_lower in ('tools', 'list tools', 'capabilities'):
        tools = mcp_client.get_available_tools()
        print(f'\n🔧 Available MCP Tools ({len(tools)}):')
        for tool in tools:
            fn = tool.get('function', {})
            print(f'  • {fn.get("name", "?")} – {fn.get("description", "")[:_TOOL_DESC_DISPLAY_LEN]}')
        print()
        return True

    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    memory = ConversationMemory(
        memory_file=CONFIG['MEMORY_FILE'],
        max_history=CONFIG['MAX_HISTORY'],
    )
    audio_input = AudioInput(
        vad_aggressiveness=CONFIG['VAD_AGGRESSIVENESS'],
        silence_timeout=CONFIG['SILENCE_TIMEOUT'],
    )
    mcp_client = init_mcp()

    startup_banner(memory, mcp_client)

    try:
        while True:
            print('─' * 70)
            print(f'👂 Waiting for wake word: say "{CONFIG["WAKE_WORD"].capitalize()}"…')

            # ── 1. Capture audio ──────────────────────────────────────────
            pcm_data = audio_input.listen()

            # ── 2. Transcribe ─────────────────────────────────────────────
            print('⏳ Processing audio…')
            text = recognize_speech(pcm_data, sample_rate=AudioInput.SAMPLE_RATE)

            if not text:
                print('❌ Could not understand audio. Try again.\n')
                continue

            print(f'✅ You said: {text}')

            # ── 3. Wake word check ────────────────────────────────────────
            if not listen_for_wake_word(text, CONFIG['WAKE_WORD']):
                wake_word_hint = CONFIG['WAKE_WORD'].capitalize()
                print(f'   (No wake word detected – say "{wake_word_hint}" to activate)\n')
                continue

            print('✅ Wake word detected!')

            command = strip_wake_word(text, CONFIG['WAKE_WORD']).strip()

            if not command:
                print('🎤 Listening for your question…\n')
                pcm_data = audio_input.listen()
                print('⏳ Processing audio…')
                command = recognize_speech(pcm_data, sample_rate=AudioInput.SAMPLE_RATE)
                if not command:
                    print('❌ Could not understand. Try again.\n')
                    continue
                print(f'✅ You said: {command}\n')

            # ── 4. Special commands ───────────────────────────────────────
            if handle_special_commands(command, memory, mcp_client):
                continue

            # ── 5. Get AI response (with MCP tool calling) ────────────────
            print('🧠 Thinking…\n')
            response = process_input(
                user_input=command,
                conversation_history=memory.get_recent_messages(),
                api_key=CONFIG['NVIDIA_API_KEY'],
                model=CONFIG['NVIDIA_LLM_MODEL'],
                temperature=CONFIG['TEMPERATURE'],
                max_tokens=CONFIG['MAX_TOKENS'],
                timeout=CONFIG['REQUEST_TIMEOUT'],
                mcp_client=mcp_client,
                max_tool_iterations=TOOLS_CONFIG['MAX_TOOL_ITERATIONS'],
            )

            print(f'🤖 JARVIS: {response}\n')

            # ── 6. Speak ──────────────────────────────────────────────────
            if CONFIG['VOICE_ENABLED']:
                print('🔊 Speaking…')
                speak(
                    response,
                    elevenlabs_api_key=CONFIG['ELEVENLABS_API_KEY'],
                    elevenlabs_voice_id=CONFIG['ELEVENLABS_VOICE_ID'],
                    elevenlabs_model=CONFIG['ELEVENLABS_MODEL'],
                )

            # ── 7. Save to memory ─────────────────────────────────────────
            memory.add_conversation(command, response)
            print(f'💾 Memory: {memory.summary()}\n')

    except KeyboardInterrupt:
        print('\n\n🛑 JARVIS shutting down. Goodbye! 👋\n')
    finally:
        audio_input.close()


if __name__ == '__main__':
    main()
