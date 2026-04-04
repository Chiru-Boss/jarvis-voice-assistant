import time
import sounddevice as sd
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
    # webrtcvad requires 10, 20, or 30 ms frames.  320 samples = 20 ms @ 16 kHz.
    FRAME_SAMPLES = 320
    FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit = 2 bytes per sample

    def __init__(self, vad_aggressiveness=2, silence_timeout=5):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_timeout = silence_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen(self, min_duration=2.0, max_duration=30.0):
        """Block until speech is detected, then record until silence.

        At least *min_duration* seconds of audio are captured once speech
        begins, so very short utterances (e.g. "Jarvis") are not truncated
        before being sent to the speech recogniser.  If no speech is detected
        within *max_duration* seconds the function returns the (possibly
        empty) audio collected so far.

        Parameters
        ----------
        min_duration : float
            Minimum seconds of audio to capture after speech starts (default 2).
        max_duration : float
            Maximum seconds to wait for/record speech before giving up (default 30).

        Returns
        -------
        bytes
            Raw PCM audio (16-bit signed, mono, 16 kHz).
        """
        frames = []
        speech_started = False
        speech_start_time = None
        silence_start = None
        listen_start = time.time()

        with sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype='int16',
            blocksize=self.FRAME_SAMPLES,
        ) as stream:
            while True:
                # Absolute timeout guard – prevents an infinite loop when the
                # microphone is silent the entire time.
                if time.time() - listen_start >= max_duration:
                    print('⚠️  Max recording duration reached.')
                    break

                data, _ = stream.read(self.FRAME_SAMPLES)
                data = bytes(data)

                # webrtcvad needs exactly FRAME_BYTES; skip malformed frames.
                if len(data) != self.FRAME_BYTES:
                    continue

                is_speech = self._is_speech(data)

                if is_speech:
                    frames.append(data)
                    if not speech_started:
                        speech_started = True
                        speech_start_time = time.time()
                        print('🗣️  Speech detected, recording…')
                    silence_start = None
                else:
                    if speech_started:
                        frames.append(data)
                        if silence_start is None:
                            silence_start = time.time()
                        else:
                            elapsed_speech = time.time() - speech_start_time
                            silence_elapsed = time.time() - silence_start
                            # Honour min_duration: do not cut off before the
                            # user has had a chance to finish a short word.
                            if (elapsed_speech >= min_duration
                                    and silence_elapsed >= self.silence_timeout):
                                break

        duration = len(frames) * self.FRAME_SAMPLES / self.SAMPLE_RATE
        print(f'✅ Captured {len(frames) * self.FRAME_BYTES} bytes '
              f'({duration:.2f} s) of audio.')
        return b''.join(frames)

    def _is_speech(self, frame):
        try:
            return self.vad.is_speech(frame, self.SAMPLE_RATE)
        except Exception as exc:
            print(f"⚠️  VAD error (frame skipped): {exc}")
            return False

    def close(self):
        pass  # No explicit cleanup needed for sounddevice


# Example usage
if __name__ == '__main__':
    audio_input = AudioInput()
    print("🎙️  Listening… speak now, recording stops on silence.")
    audio = audio_input.listen()
    print(f"✅ Captured {len(audio)} bytes of audio.")
    audio_input.close()
