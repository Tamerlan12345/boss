
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

class TestSpeakerIdRegistration(unittest.TestCase):
    def setUp(self):
        # Create mocks for dependencies
        self.mock_numpy = MagicMock()
        self.mock_numpy.linalg.norm.return_value = 1.0
        self.mock_numpy.zeros.return_value = []
        self.mock_numpy.array.return_value = []
        self.mock_numpy.float32 = float

        self.mock_resemblyzer = MagicMock()
        self.mock_resemblyzer.VoiceEncoder = MagicMock
        self.mock_preprocess_wav = MagicMock()
        self.mock_resemblyzer.preprocess_wav = self.mock_preprocess_wav

        self.mock_torchaudio = MagicMock()
        self.mock_soundfile = MagicMock()

        # Patch sys.modules to inject mocks
        # We need to include 'app' and other potential imports if they are touched
        self.modules_patcher = patch.dict(sys.modules, {
            'numpy': self.mock_numpy,
            'resemblyzer': self.mock_resemblyzer,
            'torchaudio': self.mock_torchaudio,
            'soundfile': self.mock_soundfile
        })
        self.modules_patcher.start()

        # Add app to path if not already there
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if app_path not in sys.path:
            sys.path.append(app_path)

        # Force reload/import of app.speaker_id to ensure it picks up the mocks
        if 'app.speaker_id' in sys.modules:
            del sys.modules['app.speaker_id']

        import app.speaker_id
        self.speaker_id_module = app.speaker_id
        self.SpeakerIdentifier = self.speaker_id_module.SpeakerIdentifier

    def tearDown(self):
        # Stop patcher to restore sys.modules
        self.modules_patcher.stop()

        # Clean up app.speaker_id from sys.modules to avoid pollution
        if 'app.speaker_id' in sys.modules:
            del sys.modules['app.speaker_id']

    def test_register_speaker_error_handling(self):
        """
        Verify that exceptions raised during audio preprocessing (e.g. FileNotFound, format error)
        are caught and logged, preventing application crash.
        """
        # We need to verify logging, so we patch the logger on the imported module
        with patch.object(self.speaker_id_module, 'logger') as mock_logger:
            # Setup
            encoder_mock = MagicMock()
            identifier = self.SpeakerIdentifier(encoder=encoder_mock)

            # Configure the mock preprocess_wav to raise an exception
            error_msg = "Invalid audio file format"
            # Note: app.speaker_id imports preprocess_wav from resembling
            # Since we mocked resemblyzer, app.speaker_id.preprocess_wav IS self.mock_preprocess_wav
            self.mock_preprocess_wav.side_effect = Exception(error_msg)

            # Action
            try:
                identifier.register_speaker("test_user_error", audio_file_path="bad_file.wav")
            except Exception as e:
                self.fail(f"register_speaker raised exception unexpectedly: {e}")

            # Verification
            # 1. Ensure preprocess_wav was called
            self.mock_preprocess_wav.assert_called_with("bad_file.wav")

            # 2. Ensure logger.error was called with the exception message
            mock_logger.error.assert_called_once()
            args, _ = mock_logger.error.call_args
            # The log message format is: f"Error registering speaker {name}: {e}"
            expected_log_part = f"Error registering speaker test_user_error: {error_msg}"
            self.assertIn(expected_log_part, args[0])

    def test_register_speaker_embedding_error(self):
        """
        Verify that exceptions raised during embedding generation are caught and logged.
        """
        with patch.object(self.speaker_id_module, 'logger') as mock_logger:
            # Setup
            encoder_mock = MagicMock()
            # Mock embed_utterance to raise an exception
            error_msg = "Embedding generation failed"
            encoder_mock.embed_utterance.side_effect = Exception(error_msg)

            identifier = self.SpeakerIdentifier(encoder=encoder_mock)

            # Ensure preprocess_wav returns something valid so execution reaches embed_utterance
            self.mock_preprocess_wav.return_value = MagicMock()
            self.mock_preprocess_wav.side_effect = None # Clear previous side_effect if any

            # Action
            try:
                identifier.register_speaker("test_user_embed_error", audio_file_path="valid_file.wav")
            except Exception as e:
                self.fail(f"register_speaker raised exception unexpectedly: {e}")

            # Verification
            # 1. Ensure preprocess_wav was called
            self.mock_preprocess_wav.assert_called_with("valid_file.wav")

            # 2. Ensure embed_utterance was called
            encoder_mock.embed_utterance.assert_called_once()

            # 3. Ensure logger.error was called with the exception message
            mock_logger.error.assert_called_once()
            args, _ = mock_logger.error.call_args
            expected_log_part = f"Error registering speaker test_user_embed_error: {error_msg}"
            self.assertIn(expected_log_part, args[0])

if __name__ == '__main__':
    unittest.main()
