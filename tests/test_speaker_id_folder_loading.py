
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

class TestSpeakerIdFolderLoading(unittest.TestCase):
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

    @patch('os.path.exists')
    @patch('glob.glob')
    def test_load_speakers_from_folder_no_encoder(self, mock_glob, mock_exists):
        """Test that load_speakers_from_folder returns early if encoder is None."""
        identifier = self.SpeakerIdentifier(encoder=None)
        # Manually set encoder to None because __init__ might try to create one if not provided
        identifier.encoder = None

        identifier.load_speakers_from_folder("some/path")

        mock_exists.assert_not_called()
        mock_glob.assert_not_called()

    @patch('os.path.exists')
    @patch('glob.glob')
    def test_load_speakers_from_folder_not_exists(self, mock_glob, mock_exists):
        """Test that load_speakers_from_folder returns early if folder doesn't exist."""
        mock_exists.return_value = False
        encoder_mock = MagicMock()
        identifier = self.SpeakerIdentifier(encoder=encoder_mock)

        identifier.load_speakers_from_folder("non_existent_path")

        mock_exists.assert_called_with("non_existent_path")
        mock_glob.assert_not_called()

    @patch('os.path.exists')
    @patch('glob.glob')
    def test_load_speakers_from_folder_empty(self, mock_glob, mock_exists):
        """Test that load_speakers_from_folder handles empty folder."""
        mock_exists.return_value = True
        mock_glob.return_value = []
        encoder_mock = MagicMock()
        identifier = self.SpeakerIdentifier(encoder=encoder_mock)

        with patch.object(identifier, 'register_speaker') as mock_register:
            identifier.load_speakers_from_folder("empty_folder")

            mock_exists.assert_called_with("empty_folder")
            mock_glob.assert_called()
            mock_register.assert_not_called()

    @patch('os.path.exists')
    @patch('glob.glob')
    def test_load_speakers_from_folder_invalid_extensions(self, mock_glob, mock_exists):
        """Test that load_speakers_from_folder ignores files with invalid extensions."""
        mock_exists.return_value = True
        mock_glob.return_value = [
            "folder/info.txt",
            "folder/image.png",
            "folder/data.json"
        ]
        encoder_mock = MagicMock()
        identifier = self.SpeakerIdentifier(encoder=encoder_mock)

        with patch.object(identifier, 'register_speaker') as mock_register:
            identifier.load_speakers_from_folder("folder")

            mock_register.assert_not_called()

    @patch('os.path.exists')
    @patch('glob.glob')
    def test_load_speakers_from_folder_success(self, mock_glob, mock_exists):
        """Test that load_speakers_from_folder successfully registers speakers from valid files."""
        mock_exists.return_value = True
        mock_glob.return_value = [
            "folder/Alice.wav",
            "folder/Bob.mp3",
            "folder/Charlie.FLAC",
            "folder/Dave.m4a",
            "folder/ignore.txt"
        ]
        encoder_mock = MagicMock()
        identifier = self.SpeakerIdentifier(encoder=encoder_mock)

        with patch.object(identifier, 'register_speaker') as mock_register:
            identifier.load_speakers_from_folder("folder")

            self.assertEqual(mock_register.call_count, 4)
            mock_register.assert_any_call("Alice", audio_file_path="folder/Alice.wav")
            mock_register.assert_any_call("Bob", audio_file_path="folder/Bob.mp3")
            mock_register.assert_any_call("Charlie", audio_file_path="folder/Charlie.FLAC")
            mock_register.assert_any_call("Dave", audio_file_path="folder/Dave.m4a")

    @patch('os.path.exists')
    @patch('glob.glob')
    def test_load_speakers_from_folder_none_path(self, mock_glob, mock_exists):
        """Test that load_speakers_from_folder handles None path gracefully."""
        encoder_mock = MagicMock()
        identifier = self.SpeakerIdentifier(encoder=encoder_mock)

        # This might raise TypeError in os.path.exists if not handled
        try:
            identifier.load_speakers_from_folder(None)
        except TypeError:
            self.fail("load_speakers_from_folder raised TypeError on None path")

if __name__ == '__main__':
    unittest.main()
