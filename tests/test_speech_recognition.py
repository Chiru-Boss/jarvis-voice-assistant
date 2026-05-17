import unittest
from unittest.mock import patch

from core import speech_recognition


class TestSpeechRecognitionFallback(unittest.TestCase):
    def test_returns_none_when_package_missing(self):
        with patch.object(speech_recognition, 'sr', None):
            result = speech_recognition.recognize_speech(b'\x00\x00' * 320)
        self.assertIsNone(result)
