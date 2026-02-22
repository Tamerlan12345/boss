import numpy as np
from collections import deque
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
        self.speaker_names = []
        self.speaker_embeddings = np.zeros((0, 256), dtype=np.float32)
        self._update_matrix()

        self.buffer = deque()
        self.buffer_sample_count = 0
        self.sample_rate = 16000
        self.window_size = 1.5 # Window size for embedding extraction
        self.step_size = 0.5   # Stride
        self.check_interval = int(self.step_size * self.sample_rate)
        self.samples_processed = 0
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

            # Normalize before adding
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm

            self.speakers[name] = embedding
            self._update_matrix()
            logger.info(f"Registered speaker: {name}")
        except Exception as e:
            logger.error(f"Error registering speaker {name}: {e}")

    def _update_matrix(self):
        """
        Updates the internal matrix of speaker embeddings for vectorized operations.
        Ensures all embeddings in self.speakers are normalized.
        """
        self.speaker_names = []
        embeddings_list = []

        # Iterate over items. Note: we modify self.speakers values in place if needed.
        for name, emb in self.speakers.items():
            # Normalize embedding if not already normalized
            norm = np.linalg.norm(emb)
            if norm > 1e-6 and abs(norm - 1.0) > 1e-5:
                emb = emb / norm
                self.speakers[name] = emb

            self.speaker_names.append(name)
            embeddings_list.append(emb)

        if embeddings_list:
            self.speaker_embeddings = np.array(embeddings_list, dtype=np.float32)
        else:
            self.speaker_embeddings = np.zeros((0, 256), dtype=np.float32)

    def reset_buffer(self):
        self.buffer.clear()
        self.buffer_sample_count = 0
        self.samples_processed = 0

    def process_chunk(self, chunk_bytes: bytes) -> str:
        """
        Processes a chunk of PCM 16-bit 16kHz audio.
        Returns the speaker name if identified, else None.
        """
        if not self.encoder or not self.speakers:
            return None

        # Validate chunk length for int16 (must be even)
        if len(chunk_bytes) % 2 != 0:
            return None

        # Convert bytes to float32
        int16_data = np.frombuffer(chunk_bytes, dtype=np.int16)
        float32_data = int16_data.astype(np.float32) / 32768.0

        chunk_len = len(float32_data)
        self.buffer.append(float32_data)
        self.buffer_sample_count += chunk_len

        # Keep buffer manageable (e.g. max 5 seconds)
        max_samples = 5 * self.sample_rate
        while self.buffer_sample_count > max_samples:
             popped = self.buffer.popleft()
             self.buffer_sample_count -= len(popped)

        # Run identification if we have enough data since last check
        # Ideally we want a sliding window.

        required_samples = int(self.window_size * self.sample_rate)
        self.samples_processed += chunk_len

        if self.buffer_sample_count >= required_samples and self.samples_processed >= self.check_interval:
            # Reconstruct the last window_size samples
            # Collect chunks from right to left until we have enough
            segment_chunks = []
            collected_samples = 0
            # Iterate reversed over the deque
            for chunk in reversed(self.buffer):
                segment_chunks.append(chunk)
                collected_samples += len(chunk)
                if collected_samples >= required_samples:
                    break

            # Since we collected from right to left, we need to reverse the list of chunks
            segment_chunks.reverse()

            # Concatenate
            full_segment = np.concatenate(segment_chunks)

            # Take the exact last required_samples
            segment = full_segment[-required_samples:]

            # We optimize by not running every single chunk, maybe every 0.5s worth of chunks?
            # Now we implement the optimization using check_interval (stride)
            self.samples_processed = 0

            embedding = self.encoder.embed_utterance(segment)

            # Normalize input embedding
            norm = np.linalg.norm(embedding)
            if norm < 1e-6:
                return None
            embedding = embedding / norm

            if len(self.speaker_embeddings) == 0:
                return None

            # Vectorized cosine similarity
            # self.speaker_embeddings is (N, D), embedding is (D,)
            # Result is (N,)
            similarities = np.dot(self.speaker_embeddings, embedding)

            best_idx = np.argmax(similarities)
            max_similarity = similarities[best_idx]

            if max_similarity > 0.65: # Threshold (tuned experimentally)
                return self.speaker_names[best_idx]

        return None
