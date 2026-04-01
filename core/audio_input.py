import pyaudio
import webrtcvad

class AudioInput:
    def __init__(self):
        self.vad = webrtcvad.Vad()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(
            rate=16000,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=1024
        )

    def listen(self):
        while True:
            data = self.stream.read(1024)
            is_speech = self.vad.is_speech(data, 16000)
            if is_speech:
                yield data

# Example usage
if __name__ == '__main__':
    audio_input = AudioInput()
    for audio_chunk in audio_input.listen():
        print(audio_chunk)