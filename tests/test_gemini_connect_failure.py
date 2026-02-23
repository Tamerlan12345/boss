import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import asyncio
import importlib

# DO NOT import app.gemini here at top level to avoid ModuleNotFoundError
# and to ensure we can patch sys.modules properly.

class TestGeminiConnectFailure(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Ensure clean state for imports
        if "app.gemini" in sys.modules:
            del sys.modules["app.gemini"]
        # We also want to make sure 'websockets' is not in sys.modules
        # (though it shouldn't be in this environment)
        if "websockets" in sys.modules:
            del sys.modules["websockets"]

    def tearDown(self):
        # Cleanup to avoid polluting other tests
        if "app.gemini" in sys.modules:
            del sys.modules["app.gemini"]
        if "websockets" in sys.modules:
            del sys.modules["websockets"]

    async def test_connect_failure_reraises_exception(self):
        """Test that connect() re-raises exception when websockets.connect fails."""

        # Create a mock for the websockets module
        mock_websockets = MagicMock()

        # Define the async side effect for connect
        async def mock_connect_fail(*args, **kwargs):
            raise Exception("Simulated Connection Error")

        # Apply the side effect to the connect method of the mock module
        mock_websockets.connect.side_effect = mock_connect_fail

        # Patch sys.modules to include our mock websockets
        with patch.dict(sys.modules, {"websockets": mock_websockets}):
            # Now we can safely import app.gemini
            import app.gemini
            # Reload to ensure it uses the mocked websockets
            importlib.reload(app.gemini)

            GeminiClient = app.gemini.GeminiClient

            # Mock environment variable for API key
            with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_test_key"}):
                client = GeminiClient()

                # Verify that the exception is re-raised
                with self.assertRaises(Exception) as context:
                    await client.connect()

                self.assertIn("Simulated Connection Error", str(context.exception))

    async def test_connect_logs_error(self):
        """Test that connect() logs the error before re-raising."""

        mock_websockets = MagicMock()
        async def mock_connect_fail(*args, **kwargs):
            raise Exception("Network Unreachable")
        mock_websockets.connect.side_effect = mock_connect_fail

        with patch.dict(sys.modules, {"websockets": mock_websockets}):
            import app.gemini
            importlib.reload(app.gemini)
            GeminiClient = app.gemini.GeminiClient

            with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_test_key"}):
                client = GeminiClient()

                # Check logs
                with self.assertLogs("app.gemini", level="ERROR") as cm:
                    with self.assertRaises(Exception):
                        await client.connect()

                    self.assertTrue(any("Failed to connect to Gemini: Network Unreachable" in log for log in cm.output))

if __name__ == "__main__":
    unittest.main()
