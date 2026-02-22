import unittest
from unittest.mock import patch
import os

# Set env vars before import
os.environ["SIMLI_API_KEY"] = "test_simli_key"
os.environ["SIMLI_FACE_ID"] = "test_face_id"

# Patch heavy deps during import
with patch("resemblyzer.VoiceEncoder"), \
     patch("app.speaker_id.SpeakerIdentifier"):
    from app.main import app

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

if __name__ == '__main__':
    unittest.main()
