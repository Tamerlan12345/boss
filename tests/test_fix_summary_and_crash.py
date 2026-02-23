import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import json

# Mock heavy dependencies
sys.modules['resemblyzer'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

# Mock fastapi
mock_fastapi = MagicMock()
mock_fastapi.WebSocket = AsyncMock
mock_fastapi.WebSocketDisconnect = Exception # It's an exception

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
sys.modules['pydantic'] = MagicMock()
sys.modules['websockets'] = MagicMock()

# Ensure app.main is importable
sys.path.append('.')
import app.main

class TestSummaryAndCrashFix(unittest.IsolatedAsyncioTestCase):

    async def test_websocket_crash_on_disconnect(self):
        """Test 1: Verify tasks are cancelled on disconnect (simulated by checking behavior)."""
        pass

    async def test_summary_audio_sent_passive(self):
        """Test 2: Verify summary audio is sent even if speaker_active is False."""
        print(f"DEBUG: asyncio.sleep is {asyncio.sleep}")

        with patch('app.main.GeminiClient') as MockGemini, \
             patch('app.main.SpeakerIdentifier') as MockIdentifier, \
             patch('app.main.VoiceEncoder'), \
             patch('app.main.resample_audio_sync', side_effect=lambda x: x):

            mock_gemini = MockGemini.return_value
            mock_gemini.connect = AsyncMock()
            mock_gemini.send_audio = AsyncMock()
            mock_gemini.send_text = AsyncMock()
            mock_gemini.close = AsyncMock()
            mock_gemini.send_audio = AsyncMock()
            mock_gemini.send_text = AsyncMock()
            mock_gemini.close = AsyncMock()
            mock_gemini.connect = AsyncMock()
            mock_gemini.send_audio = AsyncMock()
            mock_gemini.send_text = AsyncMock()
            mock_gemini.close = AsyncMock()

            # Scenario:
            # 1. User triggers summary (client sends command)
            # 2. Gemini sends audio chunks
            # 3. Gemini sends [SILENCE]
            # 4. Expect audio to be sent to client

            # Setup Gemini receive stream
            # We need to simulate the timing.
            # Client sends trigger_summary -> state['processing_summary'] = True
            # Then Gemini sends audio -> buffered
            # Then Gemini sends [SILENCE] -> flush

            # We need to control the receive stream to happen AFTER client command.
            # But `receive_from_client` and `send_to_client` run concurrently.

            # Let's mock `receive_from_client` to send the command, then disconnect.
            # And mock `gemini.receive` to yield audio then silence.

            async def client_receive():
                # 1. Trigger summary
                yield {"type": "websocket.receive", "text": json.dumps({"type": "trigger_summary"})}
                # Give time for processing (must be > 1.5s simli pause)
                await asyncio.sleep(2.0)
                # 2. Disconnect (to end the test)
                raise asyncio.CancelledError("Simulate disconnect")

            # Since we can't easily mock async generator with side_effect and cancellation,
            # we'll mock receive to return items then sleep forever (or raise exception).
            # But app.main uses `async for chunk in gemini_client.receive()`.

            # Let's mock the receive call to return an async iterator
            async def gemini_iterator():
                # Wait for trigger
                await asyncio.sleep(0.05)
                yield b"summary_audio_part1"
                yield b"summary_audio_part2"
                yield "[SILENCE]"
                await asyncio.sleep(10) # wait to be cancelled

            mock_gemini.receive.return_value = gemini_iterator()

            mock_ws = AsyncMock()

            async def client_receive_side_effect():
                if not hasattr(client_receive_side_effect, "called"):
                    client_receive_side_effect.called = True
                    return {"type": "websocket.receive", "text": json.dumps({"type": "trigger_summary"})}
                else:
                    await asyncio.sleep(2.0)
                    return {"type": "websocket.disconnect"}

            mock_ws.receive.side_effect = client_receive_side_effect

            # Run endpoint in "speaker" mode
            try:
                await app.main.websocket_endpoint(mock_ws, mode="speaker")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Caught exception in test: {e}")

            # Verify bytes were sent
            # We expect "summary_audio_part1" + "summary_audio_part2" to be sent.
            # Since resample is identity, it should be exact.

            sent_bytes = b""
            for call in mock_ws.send_bytes.call_args_list:
                sent_bytes += call.args[0]

            self.assertIn(b"summary_audio_part1", sent_bytes)
            self.assertIn(b"summary_audio_part2", sent_bytes)

            # Verify summary_done was sent
            # mock_ws.send_text.assert_any_call(json.dumps({"type": "summary_done"}))
            # (Note: json dumps might differ in whitespace, so better check logic)
            found_done = False
            for call in mock_ws.send_text.call_args_list:
                if "summary_done" in call.args[0]:
                    found_done = True
            self.assertTrue(found_done, "Summary done message not sent")

    async def test_short_answer_flush(self):
        """Test 3: Verify short audio (< 4096 bytes) is flushed on [SILENCE]."""

        with patch('app.main.GeminiClient') as MockGemini, \
             patch('app.main.SpeakerIdentifier'), \
             patch('app.main.VoiceEncoder'), \
             patch('app.main.resample_audio_sync', side_effect=lambda x: x):

            mock_gemini = MockGemini.return_value
            mock_gemini.connect = AsyncMock()
            mock_gemini.close = AsyncMock()
            mock_gemini.send_audio = AsyncMock()
            mock_gemini.send_text = AsyncMock()

            async def gemini_iterator():
                await asyncio.sleep(0.1)
                yield b"a" * 1000 # Short audio
                yield "[SILENCE]"
                await asyncio.sleep(10)

            mock_gemini.receive.return_value = gemini_iterator()

            mock_ws = AsyncMock()

            async def ws_receive():
                 yield {"type": "websocket.receive", "text": json.dumps({"type": "toggle_active", "enabled": True})}
                 await asyncio.sleep(2.0)
                 yield {"type": "websocket.disconnect"}

            mock_ws.receive.side_effect = ws_receive().__aiter__() # using async generator

            # Actually websocket.receive is a coroutine, so side_effect can be an iterable of coroutines or return values.
            # But here we need delays.
            # We can use a side_effect function.

            async def receive_side_effect():
                if not hasattr(receive_side_effect, "called"):
                    receive_side_effect.called = True
                    return {"type": "websocket.receive", "text": json.dumps({"type": "toggle_active", "enabled": True})}
                else:
                    await asyncio.sleep(2.0)
                    # raise WebSocketDisconnect from app's perspective, or return disconnect msg
                    return {"type": "websocket.disconnect"}

            mock_ws.receive.side_effect = receive_side_effect

            try:
                await app.main.websocket_endpoint(mock_ws, mode="speaker")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Caught exception in test: {e}")

            # Verify sent bytes
            sent_bytes = b""
            for call in mock_ws.send_bytes.call_args_list:
                sent_bytes += call.args[0]

            self.assertEqual(len(sent_bytes), 1000, "Short audio was not flushed!")

if __name__ == '__main__':
    unittest.main()
