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


def recognize_speech(pcm_data, sample_rate=16000):
    """Convert raw PCM audio to text using Google Speech Recognition.

    Parameters
    ----------
    pcm_data : bytes
        Raw 16-bit mono PCM audio at *sample_rate* Hz.
    sample_rate : int
        Audio sample rate (default 16 000 Hz).

    Returns
    -------
    str or None
        Recognised text, or ``None`` if recognition failed.
    """
    recognizer = sr.Recognizer()
    wav_bytes = pcm_to_wav_bytes(pcm_data, sample_rate=sample_rate)

    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"❌ Speech recognition network error: {e}")
        return None
