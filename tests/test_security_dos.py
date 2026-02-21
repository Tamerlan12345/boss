
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import json
import os
import asyncio
import time

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

# Mock heavy dependencies and internal modules
sys.modules["resemblyzer"] = MagicMock()
sys.modules["torchaudio"] = MagicMock()
sys.modules["torchaudio.transforms"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["torch"] = MagicMock()

# Mock app.gemini
mock_gemini_module = MagicMock()
mock_gemini_client_class = MagicMock()
mock_gemini_instance = MagicMock()
mock_gemini_instance.connect = AsyncMock()
mock_gemini_instance.send_text = AsyncMock()
mock_gemini_instance.send_audio = AsyncMock()
mock_gemini_instance.close = AsyncMock()

async def async_gen():
    yield "[SILENCE]"
    while True:
        await asyncio.sleep(0.1)
        yield "[SILENCE]"

mock_gemini_instance.receive.side_effect = async_gen
mock_gemini_client_class.return_value = mock_gemini_instance
mock_gemini_module.GeminiClient = mock_gemini_client_class
sys.modules["app.gemini"] = mock_gemini_module

# Mock app.speaker_id
sys.modules["app.speaker_id"] = MagicMock()

# Import app.main after mocking
# Note: This test patches sys.modules globally, so it should be run in isolation
# or carefully managed if part of a larger suite.
from app.main import app

client = TestClient(app)

def test_dos_malformed_json():
    """
    Tests that sending a JSON value that is not a dictionary (e.g. an integer)
    does not crash the WebSocket handler (DoS vulnerability).
    """
    # We need to capture logs to check if error occurs or is handled gracefully
    with patch("app.main.logger") as mock_logger:
        with client.websocket_connect("/ws") as websocket:
            # 1. Send malformed JSON structure that parses to an int (not a dict)
            # send_json(123) -> sends "123" string over WS, which json.loads parses as 123 (int)
            websocket.send_json(123)

            # Allow some time for processing
            time.sleep(0.5)

            # 2. Send a valid message
            websocket.send_json({"type": "mute_toggle", "enabled": False})

            # Allow some time for processing
            time.sleep(0.5)

            # Check if the AttributeError was logged (which would indicate crash/unhandled exception)
            found_attribute_error = False
            for call in mock_logger.error.call_args_list:
                args, _ = call
                if "'int' object has no attribute 'get'" in str(args[0]):
                    found_attribute_error = True
                    break

            assert not found_attribute_error, "The AttributeError was logged, meaning the fix failed and handler crashed."

            # Check if the valid message was processed.
            found_mute_log = False
            for call in mock_logger.info.call_args_list:
                args, _ = call
                if "Mute toggle" in str(args[0]):
                    found_mute_log = True
                    break

            assert found_mute_log, "The loop stopped processing messages, so the DoS still occurred."

if __name__ == "__main__":
    test_dos_malformed_json()
    print("Test Passed")
