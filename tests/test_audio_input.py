import importlib
import struct
import sys
import types
import unittest
from unittest.mock import patch


def _load_audio_input_module():
    fake_sound_device = types.SimpleNamespace(RawInputStream=object)
    with patch.dict(sys.modules, {'sounddevice': fake_sound_device}):
        sys.modules.pop('core.audio_input', None)
        return importlib.import_module('core.audio_input')


class TestAudioInputFallback(unittest.TestCase):
    def test_falls_back_when_webrtcvad_is_unavailable(self):
        audio_input = _load_audio_input_module()
        with patch.object(audio_input, 'webrtcvad', None), patch.object(
            audio_input, '_VAD_IMPORT_ERROR', ModuleNotFoundError('webrtcvad not available')
        ):
            mic = audio_input.AudioInput()
        self.assertIsInstance(mic.vad, audio_input._EnergyVAD)

    def test_energy_vad_detects_silent_and_loud_frames(self):
        audio_input = _load_audio_input_module()
        vad = audio_input._EnergyVAD(energy_threshold=100)
        silent = b'\x00\x00' * audio_input.AudioInput.FRAME_SAMPLES
        sample_count = audio_input.AudioInput.FRAME_SAMPLES
        frame_format = '<' + ('h' * sample_count)
        loud = struct.pack(frame_format, *([1200] * sample_count))

        self.assertFalse(vad.is_speech(silent, audio_input.AudioInput.SAMPLE_RATE))
        self.assertTrue(vad.is_speech(loud, audio_input.AudioInput.SAMPLE_RATE))
