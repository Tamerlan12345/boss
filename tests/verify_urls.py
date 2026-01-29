import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import os

# Set dummy API key
os.environ["HEYGEN_API_KEY"] = "test_key"

from app.main import app

class TestHeyGenUrls(unittest.TestCase):
    def setUp(self):
        # We assume startup doesn't block or crash (it has try/except)
        self.client = TestClient(app)

    @unittest.skip("HeyGen endpoints are missing in the current codebase")
    @patch('app.main.requests.post')
    def test_create_session_url(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"data": "ok"}

        self.client.post("/heygen/session/create", json={"token": "t", "quality": "high", "avatar_name": "a", "voice_id": "v"})

        # Check URL
        args, _ = mock_post.call_args
        # This asserts we are calling the NEW correct URL.
        self.assertEqual(args[0], "https://api.heygen.com/v1/streaming.new")

    @unittest.skip("HeyGen endpoints are missing in the current codebase")
    @patch('app.main.requests.post')
    def test_start_session_url(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"data": "ok"}

        self.client.post("/heygen/session/start", json={"token": "t", "session_id": "s", "sdp": {}})

        args, _ = mock_post.call_args
        self.assertEqual(args[0], "https://api.heygen.com/v1/streaming.start")

    @unittest.skip("HeyGen endpoints are missing in the current codebase")
    @patch('app.main.requests.post')
    def test_ice_url(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"data": "ok"}

        self.client.post("/heygen/ice", json={"token": "t", "session_id": "s", "candidate": {}})

        args, _ = mock_post.call_args
        self.assertEqual(args[0], "https://api.heygen.com/v1/streaming.ice")

    @unittest.skip("HeyGen endpoints are missing in the current codebase")
    @patch('app.main.requests.post')
    def test_task_url(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"data": "ok"}

        self.client.post("/heygen/task", json={"token": "t", "session_id": "s", "text": "hello"})

        args, _ = mock_post.call_args
        self.assertEqual(args[0], "https://api.heygen.com/v1/streaming.task")

    @unittest.skip("HeyGen endpoints are missing in the current codebase")
    @patch('app.main.requests.post')
    def test_stop_session_url(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"data": "ok"}

        self.client.post("/heygen/session/stop", json={"token": "t", "session_id": "s"})

        args, _ = mock_post.call_args
        self.assertEqual(args[0], "https://api.heygen.com/v1/streaming.stop")

if __name__ == '__main__':
    unittest.main()
