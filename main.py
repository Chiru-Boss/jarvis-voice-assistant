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

Hand tracking (optional – set HAND_TRACKING_ENABLED=true in .env):
  Move index finger  → mouse pointer follows
  Pinch              → left click
  Open palm          → right click
  Thumbs up          → re-activate voice listening
  Peace sign         → take screenshot
  Fist               → pause / freeze pointer
  Swipe right/left   → scroll
"""

import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import CONFIG
from config.tools_config import TOOLS_CONFIG
from config.hand_tracking_config import HAND_TRACKING_CONFIG
from core.adaptive_agent import AdaptiveAgent
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


def init_mcp(agent: AdaptiveAgent) -> MCPClient:
    """Build the tool registry, inject the adaptive agent, start the MCP server, and return a client."""
    registry = build_registry()

    # Wire the shared agent into system_tools so its new MCP tools work.
    from tools import system_tools
    system_tools.inject_agent(agent)

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

def startup_banner(memory: ConversationMemory, mcp_client: MCPClient, hand_enabled: bool = False, agent: AdaptiveAgent = None) -> None:
    tool_count = len(mcp_client.get_available_tools())
    print('\n' + '=' * 70)
    print('         🤖 J.A.R.V.I.S 2.0 – AUTONOMOUS AI AGENT 🤖')
    print('=' * 70)
    print('\n⚡ Initializing systems…')
    print('✅ NVIDIA Llama – Online')
    print('✅ Speech Recognition – Active')
    print('✅ Text-to-Speech – Ready')
    print(f'✅ Persistent Memory – {memory.summary()}')
    print(f'✅ MCP Tool Server – {tool_count} tools registered')
    if agent:
        print('✅ Adaptive Agent – Active (pattern learning enabled)')
        print('✅ Screen Vision – Ready')
        print('✅ App Controller – Ready')
        print('✅ System Executor – Ready')
    if TOOLS_CONFIG['APPROVAL_MODE']:
        print('🔒 Approval Mode – ON (destructive tools require confirmation)')
    if hand_enabled:
        print('🖐️  Hand Tracking – ACTIVE (hologram cursor enabled)')
    print(f'\n🔑 Wake word: "{CONFIG["WAKE_WORD"].capitalize()}"')
    print('💡 Commands after wake word: exit | memory | clear | tools | patterns | predict')
    print('=' * 70 + '\n')


def handle_special_commands(text: str, memory: ConversationMemory, mcp_client: MCPClient, agent: AdaptiveAgent = None) -> bool:
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

    if text_lower in ('patterns', 'show patterns', 'learned patterns') and agent:
        summary = agent.memory.get_all_patterns()
        print('\n🧠 Learned Patterns:')
        print(f"  Top apps: {', '.join(summary.get('top_apps', [])) or 'none yet'}")
        print(f"  Top searches: {', '.join(summary.get('top_searches', [])) or 'none yet'}")
        print(f"  Recent commands: {', '.join(summary.get('recent_commands', [])[-5:]) or 'none'}")
        wf = summary.get('top_workflows', [])
        if wf:
            print(f"  Top workflow: {wf[0].get('sequence', [])} (x{wf[0].get('frequency', 0)})")
        print()
        return True

    if text_lower in ('predict', 'what next', 'predictions') and agent:
        recent = agent.memory.get_recent_commands(1)
        last_cmd = recent[-1] if recent else None
        print('\n' + agent.predictor.predict_action_text(last_command=last_cmd) + '\n')
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

    # ── Adaptive AI Agent (pattern learning + screen vision) ──────────
    print('🤖 Initialising Adaptive Agent…')
    agent = AdaptiveAgent()
    print('✅ Adaptive Agent ready.')

    mcp_client = init_mcp(agent)

    # ── Hand tracking (optional) ───────────────────────────────────────
    hand_integration = None
    hand_enabled = HAND_TRACKING_CONFIG['HAND_TRACKING_ENABLED']

    if hand_enabled:
        try:
            from core.hand_voice_integration import HandVoiceIntegration
            from utils.calibration import load_calibration

            # Screenshot callback wired to MCP tool
            def _on_screenshot():
                try:
                    mcp_client.call_tool('take_screenshot', {})
                except Exception:
                    pass

            # Apply saved calibration if available
            cal_corners = load_calibration()
            if cal_corners:
                print('📐 Calibration data loaded.')

            hand_integration = HandVoiceIntegration(
                camera_device=HAND_TRACKING_CONFIG['CAMERA_DEVICE'],
                detection_confidence=HAND_TRACKING_CONFIG['HAND_DETECTION_CONFIDENCE'],
                tracking_confidence=HAND_TRACKING_CONFIG['HAND_TRACKING_CONFIDENCE'],
                pinch_threshold=HAND_TRACKING_CONFIG['PINCH_THRESHOLD'],
                smoothing=HAND_TRACKING_CONFIG['GESTURE_SMOOTHING'],
                mouse_speed=HAND_TRACKING_CONFIG['MOUSE_SPEED'],
                click_delay=HAND_TRACKING_CONFIG['CLICK_DELAY'],
                gesture_cooldown=HAND_TRACKING_CONFIG['GESTURE_COOLDOWN'],
                show_overlay=HAND_TRACKING_CONFIG['SHOW_HAND_OVERLAY'],
                on_screenshot=_on_screenshot,
                calibration_corners=cal_corners,
            )

            hand_integration.start()
            print('🖐️  Hand tracking started.')
        except Exception as exc:
            print(f'⚠️  Hand tracking unavailable: {exc}')
            hand_integration = None
            hand_enabled = False

    startup_banner(memory, mcp_client, hand_enabled=hand_enabled, agent=agent)

    try:
        while True:
            print('─' * 70)
            gesture_hint = ''
            if hand_integration and hand_integration.is_running:
                g = hand_integration.get_latest_gesture()
                if g and g not in ('none', 'point'):
                    gesture_hint = f' [Hand: {g}]'
            print(f'👂 Waiting for wake word: say "{CONFIG["WAKE_WORD"].capitalize()}"…{gesture_hint}')

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
            if handle_special_commands(command, memory, mcp_client, agent=agent):
                continue

            # ── 5. Enrich command with gesture context ────────────────────
            if hand_integration and hand_integration.is_running:
                latest_gesture = hand_integration.get_latest_gesture()
                if latest_gesture not in ('none', 'point', ''):
                    command = f'{command} [hand gesture: {latest_gesture}]'
                    print(f'🖐️  Combined with gesture: {latest_gesture}')
                    # Feed gesture into pattern learning.
                    agent.learn_from_gesture(latest_gesture)

            # ── 6. Adaptive agent pre-processing ─────────────────────────
            # Run the agent's intent detection (app open/close, search, etc.)
            # so that patterns are recorded before the LLM is consulted.
            agent_result = agent.process_command(command)
            if agent_result.get('result'):
                print(f'🤖 Agent action: {agent_result["result"]}')

            # Show predictions to help the user (non-intrusive).
            predictions = agent_result.get('predictions', [])
            if predictions:
                top = predictions[0]
                pct = int(top['confidence'] * 100)
                print(f'🔮 Prediction: {top["action"]} ({pct}%) – {top["reason"]}')

            # Build pattern context for the LLM.
            pattern_hint = agent.get_pattern_summary()

            # ── 7. Get AI response (with MCP tool calling) ────────────────
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
                pattern_hint=pattern_hint,
            )

            print(f'🤖 JARVIS: {response}\n')

            # ── 8. Speak ──────────────────────────────────────────────────
            if CONFIG['VOICE_ENABLED']:
                print('🔊 Speaking…')
                speak(
                    response,
                    elevenlabs_api_key=CONFIG['ELEVENLABS_API_KEY'],
                    elevenlabs_voice_id=CONFIG['ELEVENLABS_VOICE_ID'],
                    elevenlabs_model=CONFIG['ELEVENLABS_MODEL'],
                )

            # ── 9. Save to memory ─────────────────────────────────────────
            memory.add_conversation(command, response)
            print(f'💾 Memory: {memory.summary()}\n')

    except KeyboardInterrupt:
        print('\n\n🛑 JARVIS shutting down. Goodbye! 👋\n')
    finally:
        audio_input.close()
        if hand_integration:
            hand_integration.stop()


if __name__ == '__main__':
    main()
