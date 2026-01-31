import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import json
import asyncio
import os

# Set dummy API key
os.environ["GEMINI_API_KEY"] = "dummy_key"

from app.gemini import GeminiClient

class TestGeminiConfig(unittest.TestCase):
    async def run_async(self, coro):
        return await coro

    def test_connect_sends_correct_config(self):
        client = GeminiClient()
        client.ws = AsyncMock()

        # Mock recv to return a valid success message
        client.ws.recv.return_value = '{"serverContent": {"turnComplete": true}}'

        async def run_test():
            await client._send_setup("Test Instruction")

            # Verify send was called
            client.ws.send.assert_called_once()
            args = client.ws.send.call_args[0]
            sent_msg = json.loads(args[0])

            # Check Model ID
            self.assertEqual(
                sent_msg["setup"]["model"],
                "models/gemini-2.5-flash-native-audio-preview-12-2025"
            )

            # Check Modalities
            self.assertEqual(
                sent_msg["setup"]["generationConfig"]["responseModalities"],
                ["AUDIO"]
            )

            # Check Handshake (recv called)
            client.ws.recv.assert_called_once()

        asyncio.run(run_test())

    def test_connect_fails_on_handshake_error(self):
        client = GeminiClient()
        client.ws = AsyncMock()

        # Mock recv to raise an exception
        client.ws.recv.side_effect = Exception("Invalid Argument")

        async def run_test():
            with self.assertRaises(Exception) as context:
                await client._send_setup("Test Instruction")
            self.assertIn("Invalid Argument", str(context.exception))

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
