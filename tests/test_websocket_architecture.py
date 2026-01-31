import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import json
import os
from fastapi import WebSocket, WebSocketDisconnect

# Set dummy API key before importing app.main to avoid init errors if any
os.environ["GEMINI_API_KEY"] = "dummy_key"

import app.main

class TestWebSocketArchitecture(unittest.IsolatedAsyncioTestCase):

    async def test_parallel_execution_and_pause(self):
        # Patch dependencies IN app.main
        with patch.object(app.main, 'GeminiClient') as MockGemini, \
             patch.object(app.main, 'SpeakerIdentifier') as MockIdentifier, \
             patch.object(app.main, 'VoiceEncoder') as MockEncoder, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            # Setup Mocks
            mock_gemini = MockGemini.return_value
            mock_gemini.connect = AsyncMock()
            mock_gemini.send_audio = AsyncMock()
            mock_gemini.send_text = AsyncMock()
            mock_gemini.close = AsyncMock()

            # Mock receive to return an empty async iterator
            async def empty_stream():
                if False: yield # make it a generator
            mock_gemini.receive.return_value = empty_stream()

            # Mock Identifier
            mock_identifier = MockIdentifier.return_value
            mock_identifier.process_chunk.return_value = "Speaker1"

            # Setup WebSocket behavior
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.receive.side_effect = [
                {"type": "websocket.receive", "bytes": b"audio_chunk"},
                {"type": "websocket.disconnect"}
            ]

            # Run endpoint
            await app.main.websocket_endpoint(mock_ws)

            # Check that sleep (Simli pause) was called
            # Note: asyncio.sleep might be called multiple times if the code uses it elsewhere,
            # but we definitely expect 1.5s
            mock_sleep.assert_any_call(1.5)

            # Check that send_audio was called (proving receive loop ran)
            mock_gemini.send_audio.assert_called_with(b"audio_chunk")

            # Check for welcome prompt
            args_list = mock_gemini.send_text.call_args_list
            found_prompt = any("Generate audio immediately" in args[0][0] for args in args_list)
            self.assertTrue(found_prompt, "Welcome trigger prompt not sent")

    async def test_json_encoding_ascii(self):
         with patch.object(app.main, 'GeminiClient') as MockGemini, \
             patch.object(app.main, 'SpeakerIdentifier') as MockIdentifier, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            mock_gemini = MockGemini.return_value
            mock_gemini.connect = AsyncMock()
            mock_gemini.close = AsyncMock()
            mock_gemini.send_text = AsyncMock()

            # Simulate Gemini sending text with Cyrillic
            async def gemini_stream():
                yield "Привет"

            mock_gemini.receive.return_value = gemini_stream()

            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.receive.side_effect = [{"type": "websocket.disconnect"}]

            await app.main.websocket_endpoint(mock_ws)

            found_log = False
            for call in mock_ws.send_text.call_args_list:
                sent_str = call[0][0]
                if "type" in sent_str and "log" in sent_str:
                    # Check if the string matches default json dump (ASCII safe)
                    # "Привет" -> "\u041f\u0440..."
                    # So the string 'Привет' should NOT appear literally in sent_str
                    if "Привет" in sent_str:
                         self.fail(f"Found literal Cyrillic in JSON, expected escaped: {sent_str}")

                    data = json.loads(sent_str)
                    if data.get("text") == "Привет":
                        found_log = True

            self.assertTrue(found_log, "Did not find the log message")

if __name__ == '__main__':
    unittest.main()
