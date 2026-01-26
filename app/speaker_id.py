import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import logging
import io
import os
import glob
import soundfile as sf

logger = logging.getLogger(__name__)

class SpeakerIdentifier:
    def __init__(self, encoder=None, speakers=None):
        self.speakers = speakers.copy() if speakers else {} # name -> embedding
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = 16000
        self.window_size = 1.5 # Window size for embedding extraction
        self.step_size = 0.5   # Stride
        self.last_identification = None

        if encoder:
            self.encoder = encoder
        else:
            try:
                logger.info("Loading VoiceEncoder Model...")
                self.encoder = VoiceEncoder()
                logger.info("VoiceEncoder loaded.")
            except Exception as e:
                logger.error(f"Failed to initialize VoiceEncoder: {e}")
                self.encoder = None

    def load_speakers_from_folder(self, folder_path: str):
        if not self.encoder or not os.path.exists(folder_path):
            return

        # Expects files like "Name.wav" or "Name.mp3"
        for file_path in glob.glob(os.path.join(folder_path, "*")):
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            if ext.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                self.register_speaker(name, audio_file_path=file_path)

    def register_speaker(self, name: str, audio_file_path: str = None, audio_data: np.ndarray = None):
        if not self.encoder:
            return

        try:
            if audio_file_path:
                wav = preprocess_wav(audio_file_path)
            elif audio_data is not None:
                wav = audio_data
            else:
                return

            embedding = self.encoder.embed_utterance(wav)
            self.speakers[name] = embedding
            logger.info(f"Registered speaker: {name}")
        except Exception as e:
            logger.error(f"Error registering speaker {name}: {e}")

    def process_chunk(self, chunk_bytes: bytes) -> str:
        """
        Processes a chunk of PCM 16-bit 16kHz audio.
        Returns the speaker name if identified, else None.
        """
        if not self.encoder or not self.speakers:
            return None

        # Convert bytes to float32
        int16_data = np.frombuffer(chunk_bytes, dtype=np.int16)
        float32_data = int16_data.astype(np.float32) / 32768.0

        self.buffer = np.concatenate((self.buffer, float32_data))

        # Keep buffer manageable (e.g. max 5 seconds)
        if len(self.buffer) > 5 * self.sample_rate:
             self.buffer = self.buffer[-5*self.sample_rate:]

        # Run identification if we have enough data since last check
        # Ideally we want a sliding window.

        required_samples = int(self.window_size * self.sample_rate)

        if len(self.buffer) >= required_samples:
            # Take the last window
            segment = self.buffer[-required_samples:]

            # We optimize by not running every single chunk, maybe every 0.5s worth of chunks?
            # But the caller calls this frequently.
            # For simplicity, let's just try to run it on the segment.
            # VoiceEncoder is reasonably fast on CPU.

            # To avoid spamming, we can throttle here, but let's just return the result
            # and let the caller handle change detection.

            embedding = self.encoder.embed_utterance(segment)

            best_name = None
            max_similarity = 0

            for name, spk_emb in self.speakers.items():
                # Cosine similarity
                similarity = np.dot(embedding, spk_emb) / (np.linalg.norm(embedding) * np.linalg.norm(spk_emb))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_name = name

            if max_similarity > 0.65: # Threshold (tuned experimentally)
                return best_name

        return None
