import time
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Mock dependencies before importing
sys.modules['resemblyzer'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['soundfile'] = MagicMock()

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.speaker_id import SpeakerIdentifier

def benchmark():
    encoder = MagicMock()
    # Ensure consume_audio doesn't exit early
    speakers = {"test": np.zeros(256)}
    identifier = SpeakerIdentifier(encoder=encoder, speakers=speakers)

    # 20ms chunk at 16000Hz = 320 samples
    chunk_len_samples = 320
    chunk_len_bytes = chunk_len_samples * 2 # int16 = 2 bytes
    chunk = bytes(bytearray(os.urandom(chunk_len_bytes)))

    # Warmup
    for _ in range(100):
        identifier.consume_audio(chunk)

    identifier.reset_buffer()

    start_time = time.time()
    iterations = 10000
    for _ in range(iterations):
        identifier.consume_audio(chunk)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Time for {iterations} iterations: {duration:.4f} seconds")
    print(f"Time per iteration: {duration/iterations*1000:.4f} ms")

if __name__ == "__main__":
    benchmark()
