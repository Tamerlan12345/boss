import unittest
from unittest.mock import patch, MagicMock
import os
import sys

from fastapi.testclient import TestClient

class TestSimli(unittest.TestCase):
    def setUp(self):
        # Mock dependencies in sys.modules using patch.dict to avoid global pollution
        self.modules_patcher = patch.dict(sys.modules, {
            'resemblyzer': MagicMock(),
            'torchaudio': MagicMock(),
            'torch': MagicMock(),
            'numpy': MagicMock(),
            'soundfile': MagicMock()
        })
        self.modules_patcher.start()

        # Import app.main inside the patched environment
        # We must reload app.main if it was already imported to ensure it uses the mocks
        if 'app.main' in sys.modules:
            del sys.modules['app.main']
        from app.main import app
        self.app = app

        # Patch internal dependencies (SpeakerIdentifier, VoiceEncoder)
        self.p1 = patch("app.main.SpeakerIdentifier")
        self.p2 = patch("app.main.VoiceEncoder")
        self.mock_identifier = self.p1.start()
        self.mock_encoder = self.p2.start()

        self.client = TestClient(self.app)

    def tearDown(self):
        self.p1.stop()
        self.p2.stop()
        self.modules_patcher.stop()

    def test_config_endpoint(self):
        with patch.dict(os.environ, {"SIMLI_API_KEY": "test_simli_key", "SIMLI_FACE_ID": "test_face_id"}):
            response = self.client.get("/simli/config")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["apiKey"], "test_simli_key")
            self.assertEqual(data["faceID"], "test_face_id")

    def test_config_endpoint_missing(self):
        with patch.dict(os.environ):
            if "SIMLI_API_KEY" in os.environ: del os.environ["SIMLI_API_KEY"]
            if "SIMLI_FACE_ID" in os.environ: del os.environ["SIMLI_FACE_ID"]

            response = self.client.get("/simli/config")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIsNone(data["apiKey"])
            self.assertIsNone(data["faceID"])

if __name__ == '__main__':
    unittest.main()
