import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import json
import os
import asyncio

# Mock websockets before importing app.gemini
sys.modules["websockets"] = MagicMock()

from app.gemini import GeminiClient

class TestGeminiReceiveError(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patcher = patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"})
        self.patcher.start()

    async def asyncTearDown(self):
        self.patcher.stop()

    async def test_receive_malformed_json(self):
        """Test that receive loop continues after a JSON parse error."""
        client = GeminiClient()
        # Mock client.ws
        client.ws = AsyncMock()

        # Define the sequence of messages
        messages = [
            json.dumps({"serverContent": {"modelTurn": {"parts": [{"text": "Hello"}]}}}),
            "{bad json",
            json.dumps({"serverContent": {"modelTurn": {"parts": [{"text": "World"}]}}})
        ]

        # Mocking __aiter__ to return an iterator of messages
        client.ws.__aiter__.return_value = messages

        received_items = []

        # Patch the logger in app.gemini
        with patch("app.gemini.logger") as mock_logger:
            async for item in client.receive():
                received_items.append(item)

            # Assertions
            self.assertEqual(received_items, ["Hello", "World"])

            # Verify error log
            # We expect at least one error log call
            self.assertTrue(mock_logger.error.called)

            # Check the message of the error log
            # The exact message is "Error parsing message: {e}"
            # We check if any of the calls contain "Error parsing message"
            found_error_log = False
            for call in mock_logger.error.call_args_list:
                args, _ = call
                if "Error parsing message" in args[0]:
                    found_error_log = True
                    break

            self.assertTrue(found_error_log, "Did not find expected error log message")

if __name__ == "__main__":
    unittest.main()
