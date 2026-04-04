"""JARVIS – Professional AI Voice Assistant.

Start the assistant:
    python main.py

Wake word: say "Jarvis" to activate, then speak your question.
The assistant stops listening automatically when you finish speaking.

Voice commands (after wake word):
  exit / quit / bye / shutdown  – shut down JARVIS
  memory / history / status     – display memory summary
  clear / forget / reset        – erase conversation memory
"""

import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import CONFIG
from core.audio_input import AudioInput
from core.speech_recognition import recognize_speech
from core.wake_word import listen_for_wake_word, strip_wake_word
from core.llm_brain import process_input
from core.text_to_speech import speak
from utils.memory import ConversationMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def startup_banner(memory):
    print('\n' + '=' * 70)
    print('         🤖 J.A.R.V.I.S – PROFESSIONAL AI ASSISTANT 🤖')
    print('=' * 70)
    print('\n⚡ Initializing systems…')
    print('✅ NVIDIA Llama – Online')
    print('✅ Speech Recognition – Active')
    print('✅ Text-to-Speech – Ready')
    print(f'✅ Persistent Memory – {memory.summary()}')
    print(f'\n🔑 Wake word: "{CONFIG["WAKE_WORD"].capitalize()}"')
    print('💡 Commands after wake word: exit | memory | clear')
    print('=' * 70 + '\n')


def handle_special_commands(text, memory):
    """Check for built-in commands.  Return True if handled, False otherwise."""
    text_lower = text.lower().strip()

    if text_lower in ('exit', 'quit', 'bye', 'shutdown', 'close'):
        print('\n🛑 JARVIS shutting down. Goodbye! 👋\n')
        sys.exit(0)

    if text_lower in ('memory', 'history', 'status'):
        print(f'💾 Memory: {memory.summary()}\n')
        for i, conv in enumerate(memory.get_memory()[-5:], 1):
            ts = conv.get('timestamp', '')[:19]
            print(f'  [{i}] {ts}')
            print(f'       You: {conv["user_input"][:60]}')
            print(f'       JARVIS: {conv["bot_response"][:60]}')
        print()
        return True

    if text_lower in ('clear', 'forget', 'reset'):
        memory.clear_memory()
        print('💾 Memory cleared!\n')
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

    startup_banner(memory)

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

            # ── 3. Wake word check ────────────────────────────────────────
            if not listen_for_wake_word(text, CONFIG['WAKE_WORD']):
                print(f'   (Heard: "{text}" – no wake word detected)\n')
                continue

            command = strip_wake_word(text, CONFIG['WAKE_WORD']).strip()
            print(f'✅ You said: {text}\n')

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
            if handle_special_commands(command, memory):
                continue

            # ── 5. Get AI response ────────────────────────────────────────
            print('🧠 Thinking…\n')
            response = process_input(
                user_input=command,
                conversation_history=memory.get_recent_messages(),
                api_key=CONFIG['NVIDIA_API_KEY'],
                model=CONFIG['NVIDIA_LLM_MODEL'],
                temperature=CONFIG['TEMPERATURE'],
                max_tokens=CONFIG['MAX_TOKENS'],
                timeout=CONFIG['REQUEST_TIMEOUT'],
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
