import sys
from unittest.mock import MagicMock, AsyncMock

# Mock heavy/missing dependencies BEFORE importing app.main
sys.modules['resemblyzer'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['websockets'] = MagicMock()
sys.modules['pydantic'] = MagicMock()

# Mock fastapi
mock_fastapi = MagicMock()
class MockWebSocket:
    async def receive(self): pass
    async def send_text(self, data): pass
    async def send_bytes(self, data): pass
    async def accept(self): pass
    async def close(self): pass
mock_fastapi.WebSocket = MockWebSocket
mock_fastapi.WebSocketDisconnect = Exception

# Configure FastAPI app decorators to be transparent
mock_app = MagicMock()
def identity_decorator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator
mock_app.websocket.side_effect = identity_decorator
mock_app.get.side_effect = identity_decorator
mock_app.on_event.side_effect = identity_decorator
mock_fastapi.FastAPI.return_value = mock_app

sys.modules['fastapi'] = mock_fastapi
sys.modules['fastapi.staticfiles'] = MagicMock()
sys.modules['fastapi.templating'] = MagicMock()
sys.modules['fastapi.requests'] = MagicMock()

import unittest
from unittest.mock import patch
# from fastapi import WebSocket, WebSocketDisconnect # Removed real import
WebSocket = MockWebSocket
WebSocketDisconnect = Exception

import asyncio
# Make sure we can find 'app'
sys.path.append('.')
import app.main

class TestBroadExceptionFix(unittest.IsolatedAsyncioTestCase):
    async def test_exception_in_receive_loop(self):
        print("\nRunning regression test for broad exception fix...")
        original_sleep = asyncio.sleep
        # Patch dependencies IN app.main
        with patch('app.main.GeminiClient') as MockGemini, \
             patch('app.main.SpeakerIdentifier') as MockIdentifier, \
             patch('app.main.VoiceEncoder') as MockEncoder, \
             patch('app.main.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            # Configure sleep to break keep_alive loop
            # Delay the crash until enough processing has happened
            async def sleep_side_effect(duration):
                # Yield control to allow other tasks to run
                await original_sleep(0)
                if duration == 5:
                    # Check if send_audio has been called enough times (chunk1 and chunk2)
                    # mock_gemini is not fully configured yet in this scope?
                    # We can access mock_gemini variable from outer scope if defined.
                    # But mock_gemini is defined below.
                    # However, MockGemini.return_value is the instance.
                    instance = MockGemini.return_value
                    if instance.send_audio.call_count >= 2:
                         raise Exception("Stop Keep Alive")

                    # Also crash if too many iterations (safety)
                    if hasattr(sleep_side_effect, 'calls'):
                        sleep_side_effect.calls += 1
                    else:
                        sleep_side_effect.calls = 1

                    if sleep_side_effect.calls > 100:
                         raise Exception("Stop Keep Alive Timeout")

                return None
            mock_sleep.side_effect = sleep_side_effect

            mock_gemini = MockGemini.return_value
            mock_gemini.connect = AsyncMock()
            mock_gemini.send_audio = AsyncMock()
            mock_gemini.send_text = AsyncMock()
            # Mock receive to be an empty async iterator
            async def empty_stream():
                if False: yield
            mock_gemini.receive.return_value = empty_stream()
            mock_gemini.close = AsyncMock()

            mock_identifier = MockIdentifier.return_value
            # Make consume_audio raise an Exception on the first call, return segment on second
            mock_identifier.consume_audio.side_effect = [Exception("Processing Error"), "segment2"]
            mock_identifier.identify_speaker.return_value = "Speaker1"

            mock_ws = AsyncMock(spec=WebSocket)
            # Simulate:
            # 1. Message that triggers exception
            # 2. Message that should be processed if loop continues
            # 3. Disconnect
            mock_ws.receive.side_effect = [
                {"type": "websocket.receive", "bytes": b"chunk1"},
                {"type": "websocket.receive", "bytes": b"chunk2"},
                {"type": "websocket.disconnect"}
            ]

            # Run endpoint
            try:
                await app.main.websocket_endpoint(mock_ws)
            except Exception:
                pass # expected from keep_alive exception?
            # Actually app.main catches Exception inside websocket_endpoint and logs it.
            # But keep_alive exception is caught inside keep_alive too.
            # So websocket_endpoint should return cleanly.

            # Check if send_audio was called for chunk2
            calls = mock_gemini.send_audio.call_args_list
            chunk2_processed = False
            for call_obj in calls:
                if call_obj.args and call_obj.args[0] == b"chunk2":
                    chunk2_processed = True
                    break

            if not chunk2_processed:
                print("[FAIL] Regression: Loop exited after exception, chunk2 was not processed.")
            else:
                print("[PASS] Verified: Loop continued after exception, chunk2 was processed.")

            # Assert for test
            self.assertTrue(chunk2_processed, "The receive loop should continue after a processing exception.")

if __name__ == '__main__':
    unittest.main()
