import time
import struct
import pyaudio
import webrtcvad


class AudioInput:
    """Microphone capture with WebRTC VAD-based smart silence detection.

    Records until the user stops speaking (or ``silence_timeout`` seconds
    of continuous silence have elapsed after at least some speech is
    detected).  Returns the captured audio as raw PCM bytes (16-bit,
    mono, 16 kHz) suitable for passing to a speech recogniser.
    """

    SAMPLE_RATE = 16000
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    # webrtcvad requires 10, 20, or 30 ms frames.  320 samples = 20 ms @ 16 kHz.
    FRAME_SAMPLES = 320
    FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit = 2 bytes per sample

    def __init__(self, vad_aggressiveness=2, silence_timeout=5):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_timeout = silence_timeout
        self._pa = pyaudio.PyAudio()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen(self):
        """Block until speech is detected, then record until silence.

        Returns
        -------
        bytes
            Raw PCM audio (16-bit signed, mono, 16 kHz).
        """
        stream = self._pa.open(
            rate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            format=self.FORMAT,
            input=True,
            frames_per_buffer=self.FRAME_SAMPLES,
        )
        try:
            return self._record_until_silence(stream)
        finally:
            stream.stop_stream()
            stream.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_until_silence(self, stream):
        frames = []
        speech_started = False
        silence_start = None

        while True:
            data = stream.read(self.FRAME_SAMPLES, exception_on_overflow=False)

            # webrtcvad needs exactly FRAME_BYTES; skip malformed frames.
            if len(data) != self.FRAME_BYTES:
                continue

            is_speech = self._is_speech(data)

            if is_speech:
                frames.append(data)
                speech_started = True
                silence_start = None
            else:
                if speech_started:
                    frames.append(data)
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= self.silence_timeout:
                        break

        return b''.join(frames)

    def _is_speech(self, frame):
        try:
            return self.vad.is_speech(frame, self.SAMPLE_RATE)
        except Exception as exc:
            print(f"⚠️  VAD error (frame skipped): {exc}")
            return False

    def close(self):
        self._pa.terminate()


# Example usage
if __name__ == '__main__':
    audio_input = AudioInput()
    print("🎙️  Listening… speak now, recording stops on silence.")
    audio = audio_input.listen()
    print(f"✅ Captured {len(audio)} bytes of audio.")
    audio_input.close()
