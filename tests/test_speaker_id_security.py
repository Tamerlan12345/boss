
import unittest
from unittest.mock import MagicMock
import sys
import os
import numpy as np

# Mock heavy dependencies
mock_resemblyzer = MagicMock()
mock_resemblyzer.VoiceEncoder = MagicMock
mock_resemblyzer.preprocess_wav = MagicMock
sys.modules['resemblyzer'] = mock_resemblyzer

mock_torchaudio = MagicMock()
sys.modules['torchaudio'] = mock_torchaudio

mock_soundfile = MagicMock()
sys.modules['soundfile'] = mock_soundfile

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.speaker_id import SpeakerIdentifier

class TestSpeakerIdentifierSecurity(unittest.TestCase):
    def test_odd_length_chunk_dos(self):
        """
        Verify that processing an odd-length byte chunk does not raise ValueError
        and returns None gracefully.
        """
        # Setup
        encoder_mock = MagicMock()
        identifier = SpeakerIdentifier(encoder=encoder_mock)

        # We need to register a speaker so it doesn't return early due to empty speakers
        identifier.speakers = {'test': np.zeros(256)}

        # Create an odd-length chunk (3 bytes)
        odd_chunk = b'\x00\x00\x00'

        try:
            result = identifier.process_chunk(odd_chunk)
            # Should return None if chunk is invalid/skipped
            self.assertIsNone(result, "Expected None for odd-length chunk processing")
        except ValueError as e:
            self.fail(f"DoS Vulnerability: process_chunk raised ValueError for odd-length chunk: {e}")
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

if __name__ == '__main__':
    unittest.main()
