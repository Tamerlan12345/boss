import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import json
import base64
import os
import asyncio

# Mock websockets before importing app.gemini
sys.modules["websockets"] = MagicMock()

from app.gemini import GeminiClient

class TestGeminiAudio(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Set dummy API key for all tests
        self.patcher = patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"})
        self.patcher.start()

    async def asyncTearDown(self):
        self.patcher.stop()

    async def test_send_audio(self):
        """Test sending audio when connected."""
        client = GeminiClient()
        # Mock the websocket connection
        client.ws = AsyncMock()

        audio_data = b'\x01\x02\x03'
        await client.send_audio(audio_data)

        # Verify send was called once
        client.ws.send.assert_called_once()

        # Verify the message structure
        args, _ = client.ws.send.call_args
        msg = json.loads(args[0])

        self.assertIn("realtimeInput", msg)
        self.assertIn("mediaChunks", msg["realtimeInput"])
        chunks = msg["realtimeInput"]["mediaChunks"]
        self.assertEqual(len(chunks), 1)

        chunk = chunks[0]
        self.assertEqual(chunk["mimeType"], "audio/pcm;rate=16000")

        expected_b64 = base64.b64encode(audio_data).decode("utf-8")
        self.assertEqual(chunk["data"], expected_b64)

    async def test_send_audio_no_connection(self):
        """Test sending audio when not connected (should do nothing)."""
        client = GeminiClient()
        # Ensure ws is None
        client.ws = None

        audio_data = b'\x01\x02\x03'
        # Should not raise any exception
        await client.send_audio(audio_data)
