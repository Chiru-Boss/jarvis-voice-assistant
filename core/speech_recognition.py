import io
import wave
import speech_recognition as sr


def pcm_to_wav_bytes(pcm_data, sample_rate=16000, channels=1, sampwidth=2):
    """Wrap raw PCM bytes in a WAV container and return as bytes."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def recognize_speech(pcm_data, sample_rate=16000, retries=2):
    """Convert raw PCM audio to text using Google Speech Recognition.

    On transient failures (network errors or unintelligible audio) the
    recognition is retried up to *retries* additional times before giving up.

    Parameters
    ----------
    pcm_data : bytes
        Raw 16-bit mono PCM audio at *sample_rate* Hz.
    sample_rate : int
        Audio sample rate (default 16 000 Hz).
    retries : int
        Number of additional attempts if the first try fails (default 2).

    Returns
    -------
    str or None
        Recognised text, or ``None`` if all attempts failed.
    """
    if not pcm_data:
        print('⚠️  No audio data to recognise.')
        return None

    recognizer = sr.Recognizer()
    wav_bytes = pcm_to_wav_bytes(pcm_data, sample_rate=sample_rate)
    max_attempts = retries + 1

    for attempt in range(1, max_attempts + 1):
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f'🔍 Recognized: "{text}"')
            return text
        except sr.UnknownValueError:
            print(f'⚠️  Could not understand audio (attempt {attempt}/{max_attempts}).')
        except sr.RequestError as e:
            print(f'❌ Speech recognition network error (attempt {attempt}/{max_attempts}): {e}')

    return None
