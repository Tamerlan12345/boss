import unittest
from unittest.mock import patch, MagicMock
import requests
import json
import os

# Set environment variable to avoid error in startup if not present,
# although we will try to mock dependencies.
os.environ["HEYGEN_API_KEY"] = "test_key"

# We need to mock VoiceEncoder before importing app.main because it is instantiated in startup
# actually it is instantiated in startup_event, so patching it before creating TestClient might work if we use with TestClient(app)
# But `from app.main import app` will run top level code.

from fastapi.testclient import TestClient
from app.main import app

class TestHeyGenProxy(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('app.main.requests.post')
    def test_create_session_error_handling(self, mock_post):
        # Setup mock to raise HTTPError with custom response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"code": 400, "message": "invalid_voice_id"}
        mock_response.text = '{"code": 400, "message": "invalid_voice_id"}'

        # requests.exceptions.HTTPError takes args usually.
        # When raise_for_status() is called, it raises this error.
        error = requests.exceptions.HTTPError("400 Client Error: Bad Request", response=mock_response)

        # Configure mock_post to return a response that raises on raise_for_status
        # or just side_effect if the code calls raise_for_status on the result of post.
        # In app/main.py:
        # resp = requests.post(...)
        # resp.raise_for_status()

        # So mock_post should return the mock_response, and mock_response.raise_for_status should raise the error.
        mock_response.raise_for_status.side_effect = error
        mock_post.return_value = mock_response

        # Make request to our endpoint
        response = self.client.post("/heygen/session/create", json={
            "token": "fake_token",
            "quality": "medium",
            "avatar_name": "avatar_id",
            "voice_id": "voice_id"
        })

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check if the error field matches our expected JSON detail
        # Note: Before the fix, this will fail because it returns string representation of exception
        self.assertEqual(data.get("error"), {"code": 400, "message": "invalid_voice_id"})

if __name__ == '__main__':
    unittest.main()
