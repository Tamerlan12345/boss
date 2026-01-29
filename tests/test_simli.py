import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import asyncio

# Set env vars before import
os.environ["SIMLI_API_KEY"] = "test_simli_key"
os.environ["SIMLI_FACE_ID"] = "test_face_id"
os.environ["OPENAI_API_KEY"] = "test_openai_key"

# Patch heavy deps during import
with patch("resemblyzer.VoiceEncoder"), \
     patch("app.speaker_id.SpeakerIdentifier"), \
     patch("openai.AsyncOpenAI"):
    from app.main import app, text_to_speech_pcm

from fastapi.testclient import TestClient

class TestSimli(unittest.TestCase):
    def setUp(self):
        # Patch dependencies in app.main to ensure startup_event uses mocks
        self.p1 = patch("app.main.SpeakerIdentifier")
        self.p2 = patch("app.main.VoiceEncoder")
        self.mock_identifier = self.p1.start()
        self.mock_encoder = self.p2.start()

        self.client = TestClient(app)

    def tearDown(self):
        self.p1.stop()
        self.p2.stop()

    def test_config_endpoint(self):
        response = self.client.get("/simli/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["apiKey"], "test_simli_key")
        self.assertEqual(data["faceID"], "test_face_id")

    @patch("app.main.openai_client.audio.speech.create", new_callable=AsyncMock)
    @patch("app.main.resample_audio_sync")
    def test_tts_function(self, mock_resample, mock_speech_create):
        # Setup mocks
        mock_response = MagicMock()
        mock_response.content = b"fake_audio_bytes"
        mock_speech_create.return_value = mock_response

        mock_resample.return_value = b"resampled_bytes"

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(text_to_speech_pcm("Hello"))
        finally:
            loop.close()

        self.assertEqual(result, b"resampled_bytes")
        mock_speech_create.assert_called_once()
        mock_resample.assert_called_with(b"fake_audio_bytes")

if __name__ == '__main__':
    unittest.main()
