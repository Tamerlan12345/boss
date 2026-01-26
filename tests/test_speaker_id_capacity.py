import unittest
import numpy as np
import sys
import os

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.speaker_id import SpeakerIdentifier

class MockVoiceEncoder:
    def embed_utterance(self, wav):
        # Create a deterministic embedding based on the mean of the wav
        # wav is expected to be a numpy array
        val = np.mean(wav)
        # Round to nearest 0.01 to handle int16 quantization noise
        # 0.1 * 32768 = 3276.8. If 3276, 3276/32768 = 0.099975.
        # 0.1 vs 0.099975. Rounding to 2 decimal places gives 0.10.
        val_rounded = round(val, 2)
        # Create a random-looking vector seeded by val
        np.random.seed(int(val_rounded * 10000))
        # Use standard normal distribution to get vectors centered around 0
        return np.random.randn(256)

class TestSpeakerIdentifier(unittest.TestCase):
    def test_capacity_8_speakers(self):
        encoder = MockVoiceEncoder()
        identifier = SpeakerIdentifier(encoder=encoder)

        # Register 8 speakers
        for i in range(8):
            # Create a dummy audio signal.
            # We use different values to ensure different embeddings from our mock encoder
            dummy_wav = np.ones(1000) * (i + 1) * 0.1
            identifier.register_speaker(f"Speaker_{i}", audio_data=dummy_wav)

        self.assertEqual(len(identifier.speakers), 8)

        # Test identification
        for i in range(8):
            # Create a chunk that matches Speaker_i
            # The identifier buffers data. We need to feed enough data.
            # Identifier buffer size default is 5s, window is 1.5s.
            # Sample rate 16000.
            # We need at least 1.5 * 16000 samples in the buffer.

            # Create a chunk of 2 seconds
            chunk_len = int(2.0 * 16000)
            # Create raw PCM bytes. float32 -> int16 mapping.
            # Our mock encoder uses mean of float data.
            # So we create float data first.
            float_data = np.ones(chunk_len, dtype=np.float32) * (i + 1) * 0.1

            # Convert to int16 bytes
            int16_data = (float_data * 32768).astype(np.int16)
            chunk_bytes = int16_data.tobytes()

            # Process chunk
            # Note: process_chunk accumulates buffer.
            # We might need to clear buffer or just expect it to work on the last window.
            # But process_chunk appends.

            # To avoid buffer pollution from previous iterations in this test loop,
            # we can create a fresh identifier or reset buffer.
            # But let's just clear buffer manually for the test
            identifier.buffer = np.array([], dtype=np.float32)

            identified_name = identifier.process_chunk(chunk_bytes)

            self.assertEqual(identified_name, f"Speaker_{i}")

    def test_unknown_speaker(self):
        encoder = MockVoiceEncoder()
        identifier = SpeakerIdentifier(encoder=encoder)

        # Register Speaker_0
        dummy_wav = np.ones(1000) * 0.1
        identifier.register_speaker("Speaker_0", audio_data=dummy_wav)

        # Test with audio corresponding to "0.5" which is different from 0.1
        chunk_len = int(2.0 * 16000)
        float_data = np.ones(chunk_len, dtype=np.float32) * 0.5
        int16_data = (float_data * 32768).astype(np.int16)
        chunk_bytes = int16_data.tobytes()

        identifier.buffer = np.array([], dtype=np.float32)
        identified_name = identifier.process_chunk(chunk_bytes)

        # Should be None if similarity is low.
        # With our mock encoder, random vectors should have low similarity.
        self.assertIsNone(identified_name)

if __name__ == '__main__':
    unittest.main()
