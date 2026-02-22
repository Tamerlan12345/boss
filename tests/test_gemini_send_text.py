import unittest
from unittest.mock import AsyncMock
import json
import asyncio
import os

# Set dummy API key
os.environ["GEMINI_API_KEY"] = "dummy_key"

from app.gemini import GeminiClient

class TestGeminiSendText(unittest.TestCase):
    def test_send_text_valid(self):
        client = GeminiClient()
        client.ws = AsyncMock()

        async def run_test():
            await client.send_text("Hello Gemini")

            # Verify send was called
            client.ws.send.assert_called_once()
            args = client.ws.send.call_args[0]
            sent_msg = json.loads(args[0])

            # Verify JSON structure
            self.assertIn("clientContent", sent_msg)
            client_content = sent_msg["clientContent"]

            self.assertIn("turns", client_content)
            self.assertEqual(len(client_content["turns"]), 1)

            turn = client_content["turns"][0]
            self.assertEqual(turn["role"], "user")

            self.assertIn("parts", turn)
            self.assertEqual(len(turn["parts"]), 1)
            self.assertEqual(turn["parts"][0]["text"], "Hello Gemini")

            self.assertTrue(client_content["turnComplete"])

        asyncio.run(run_test())

    def test_send_text_no_ws(self):
        client = GeminiClient()
        client.ws = None # Not connected

        async def run_test():
            # Should not raise exception
            await client.send_text("Hello Gemini")

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
