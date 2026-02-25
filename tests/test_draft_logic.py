import sys
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# --- MOCKING DEPENDENCIES BEFORE IMPORT ---
sys.modules['resemblyzer'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['websockets'] = MagicMock()
sys.modules['pydantic'] = MagicMock()

# Mock FastAPI
mock_fastapi = MagicMock()
class MockWebSocket:
    async def receive(self): pass
    async def send_text(self, data): pass
    async def send_bytes(self, data): pass
    async def accept(self): pass
    async def close(self): pass

mock_fastapi.WebSocket = MockWebSocket
mock_fastapi.WebSocketDisconnect = Exception

# Configure FastAPI app decorators
mock_app = MagicMock()
def identity_decorator(*args, **kwargs):
    def decorator(func): return func
    return decorator
mock_app.websocket.side_effect = identity_decorator
mock_app.get.side_effect = identity_decorator
mock_app.on_event.side_effect = identity_decorator
mock_fastapi.FastAPI.return_value = mock_app

sys.modules['fastapi'] = mock_fastapi
sys.modules['fastapi.staticfiles'] = MagicMock()
sys.modules['fastapi.templating'] = MagicMock()
sys.modules['fastapi.requests'] = MagicMock()

# Import app.main after mocking
import app.main

class TestDraftLogic(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Reset global state
        app.main.session_manager.state["draft_mode_enabled"] = True
        app.main.session_manager.state["is_speaking_approved"] = False
        app.main.session_manager.current_draft = ""
        app.main.session_manager.last_spoken_words = ""

    async def test_draft_mode_unapproved_accumulates(self):
        """Verify that unapproved drafts accumulate text (including silence)."""
        print("\nRunning Test: Draft Mode Unapproved Accumulation")

        # Setup Mocks
        mock_ws = AsyncMock(spec=MockWebSocket)
        async def ws_receive_side_effect(): await asyncio.Future()
        mock_ws.receive.side_effect = ws_receive_side_effect

        with patch('app.main.GeminiClient') as MockGemini, \
             patch('app.main.SpeakerIdentifier') as MockIdentifier, \
             patch('app.main.resample_audio_sync') as mock_resample, \
             patch('app.main.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            # Smart sleep
            async def smart_sleep(seconds):
                if seconds == 1.5: return
                await asyncio.sleep(0.01)
            mock_sleep.side_effect = smart_sleep

            mock_gemini_instance = MockGemini.return_value
            mock_gemini_instance.connect = AsyncMock()
            mock_gemini_instance.send_audio = AsyncMock()
            mock_gemini_instance.send_text = AsyncMock()
            mock_gemini_instance.close = AsyncMock()

            # Break loop logic
            mock_gemini_instance.connect.side_effect = [None, app.main.WebSocketDisconnect()]
            MockGemini.return_value = mock_gemini_instance

            async def stream():
                yield "PLAN: test"
                yield "[SILENCE]"
            mock_gemini_instance.receive.return_value = stream()

            try:
                await app.main.websocket_endpoint(mock_ws)
            except Exception: pass

            draft_content = app.main.session_manager.current_draft
            print(f"Draft Content: '{draft_content}'")
            self.assertIn("PLAN: test", draft_content)
            self.assertIn("[SILENCE]", draft_content)

    async def test_approved_speech_skips_draft_and_resets(self):
        """Verify that approved speech is NOT drafted and resets approval state."""
        print("\nRunning Test: Approved Speech Logic")

        app.main.session_manager.state["is_speaking_approved"] = True

        mock_ws = AsyncMock(spec=MockWebSocket)
        async def ws_receive_side_effect(): await asyncio.Future()
        mock_ws.receive.side_effect = ws_receive_side_effect

        with patch('app.main.GeminiClient') as MockGemini, \
             patch('app.main.SpeakerIdentifier') as MockIdentifier, \
             patch('app.main.resample_audio_sync') as mock_resample, \
             patch('app.main.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            async def smart_sleep(seconds):
                if seconds == 1.5: return
                await asyncio.sleep(0.01)
            mock_sleep.side_effect = smart_sleep

            mock_gemini_instance = MockGemini.return_value
            mock_gemini_instance.connect = AsyncMock()
            mock_gemini_instance.send_audio = AsyncMock()
            mock_gemini_instance.send_text = AsyncMock()
            mock_gemini_instance.close = AsyncMock()

            mock_gemini_instance.connect.side_effect = [None, app.main.WebSocketDisconnect()]
            MockGemini.return_value = mock_gemini_instance

            async def stream():
                yield "Speech Content."
                yield "[SILENCE]"
            mock_gemini_instance.receive.return_value = stream()

            try:
                await app.main.websocket_endpoint(mock_ws)
            except Exception: pass

            draft_content = app.main.session_manager.current_draft
            print(f"Captured Draft Content: '{draft_content}'")

            self.assertNotIn("Speech Content.", draft_content)

            is_approved = app.main.session_manager.state["is_speaking_approved"]
            print(f"Final Approval State: {is_approved}")
            self.assertFalse(is_approved, "Approval state should reset after silence.")

if __name__ == '__main__':
    unittest.main()
