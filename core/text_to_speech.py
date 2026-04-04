import pyttsx3
import requests


def _speak_pyttsx3(text):
    """Speak *text* using the local pyttsx3 engine."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    engine.setProperty('volume', 0.9)
    engine.say(text)
    engine.runAndWait()


def speak(text, elevenlabs_api_key=None, elevenlabs_voice_id=None,
          elevenlabs_model='eleven_monolingual_v1'):
    """Convert *text* to speech.

    Attempts ElevenLabs first when credentials are provided; falls back
    to the bundled pyttsx3 engine automatically.

    Parameters
    ----------
    text : str
        The text to speak aloud.
    elevenlabs_api_key : str or None
        ElevenLabs API key.  If ``None`` or empty, ElevenLabs is skipped.
    elevenlabs_voice_id : str or None
        ElevenLabs voice ID.
    elevenlabs_model : str
        ElevenLabs TTS model identifier.
    """
    if elevenlabs_api_key and elevenlabs_voice_id:
        try:
            url = (
                f'https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}'
            )
            headers = {
                'xi-api-key': elevenlabs_api_key,
                'Content-Type': 'application/json',
            }
            payload = {
                'text': text[:500],
                'model_id': elevenlabs_model,
                'voice_settings': {'stability': 0.5, 'similarity_boost': 0.75},
            }
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            if response.status_code == 200:
                audio_file = 'jarvis_response.mp3'
                with open(audio_file, 'wb') as f:
                    f.write(response.content)
                try:
                    import subprocess
                    import sys
                    if sys.platform == 'win32':
                        import winsound
                        winsound.PlaySound(audio_file, winsound.SND_FILENAME)
                    else:
                        subprocess.run(
                            ['ffplay', '-nodisp', '-autoexit', audio_file],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    return
                except Exception as exc:
                    print(f"⚠️  Audio playback failed, falling back to pyttsx3: {exc}")
            else:
                print(
                    f"⚠️  ElevenLabs API returned {response.status_code}, "
                    "falling back to pyttsx3."
                )
        except Exception as exc:
            print(f"⚠️  ElevenLabs TTS failed, falling back to pyttsx3: {exc}")

    _speak_pyttsx3(text)
