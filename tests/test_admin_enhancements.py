import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import asyncio
import json
import os

# --- Comprehensive Mocking of Environment ---
mock_fastapi = MagicMock()
class WebSocketDisconnect(Exception): pass
mock_fastapi.WebSocketDisconnect = WebSocketDisconnect
mock_fastapi.WebSocket = MagicMock()

# Mock Decorators to return original function
def mock_decorator(*args, **kwargs):
    def wrapper(func):
        return func
    return wrapper

mock_app_instance = MagicMock()
mock_app_instance.websocket.side_effect = mock_decorator
mock_app_instance.get.side_effect = mock_decorator
mock_app_instance.post.side_effect = mock_decorator
mock_app_instance.on_event.side_effect = mock_decorator

mock_fastapi.FastAPI.return_value = mock_app_instance

sys.modules["fastapi"] = mock_fastapi
sys.modules["fastapi.staticfiles"] = MagicMock()
sys.modules["fastapi.templating"] = MagicMock()
sys.modules["fastapi.requests"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["uvicorn"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torchaudio"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["resemblyzer"] = MagicMock()
mock_gemini_module = MagicMock()
class GeminiClient:
    async def connect(self, system_instruction=None): pass
    async def send_text(self, text): pass
    async def close(self): pass
mock_gemini_module.GeminiClient = GeminiClient
sys.modules["app.gemini"] = mock_gemini_module
sys.modules["app.speaker_id"] = MagicMock()

from app.main import SessionManager, admin_websocket, session_manager
from app.prompts import SUMMARIZE_TEXT_PROMPT

class TestAdminEnhancements(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        session_manager.admin_websockets = []
        session_manager.active_client_ws = None
        session_manager.gemini_client = AsyncMock()
        session_manager.gemini_client.send_text = AsyncMock()
        session_manager.current_draft = ""
        session_manager.transcript = []

    async def test_summary_from_text(self):
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()

        text_to_summarize = "This is a test article."
        mock_ws.receive_text.side_effect = [
            json.dumps({"type": "summary_from_text", "text": text_to_summarize}),
            WebSocketDisconnect()
        ]

        try:
            await admin_websocket(mock_ws)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Exception in test_summary_from_text: {e}")

        # Check Gemini send_text
        session_manager.gemini_client.send_text.assert_called()
        call_args = session_manager.gemini_client.send_text.call_args[0][0]
        self.assertIn(SUMMARIZE_TEXT_PROMPT, call_args)
        self.assertIn(text_to_summarize, call_args)
        self.assertIn("SPEAK RUSSIAN", call_args)

    async def test_approve_russian_enforcement(self):
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()

        session_manager.current_draft = "PLAN: Test plan."

        mock_ws.receive_text.side_effect = [
            json.dumps({"type": "approve"}),
            WebSocketDisconnect()
        ]

        try:
            await admin_websocket(mock_ws)
        except WebSocketDisconnect:
            pass

        call_args = session_manager.gemini_client.send_text.call_args[0][0]
        self.assertIn("ОТВЕЧАЙ ИСКЛЮЧИТЕЛЬНО НА РУССКОМ ЯЗЫКЕ", call_args)

    async def test_manual_russian_enforcement(self):
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()

        manual_text = "Say hello."
        mock_ws.receive_text.side_effect = [
            json.dumps({"type": "manual", "text": manual_text}),
            WebSocketDisconnect()
        ]

        try:
            await admin_websocket(mock_ws)
        except WebSocketDisconnect:
            pass

        call_args = session_manager.gemini_client.send_text.call_args[0][0]
        self.assertIn("ОТВЕЧАЙ ИСКЛЮЧИТЕЛЬНО НА РУССКОМ ЯЗЫКЕ", call_args)

    async def test_trigger_summary_russian_enforcement(self):
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()

        mock_ws.receive_text.side_effect = [
            json.dumps({"type": "trigger_summary"}),
            WebSocketDisconnect()
        ]

        try:
            await admin_websocket(mock_ws)
        except WebSocketDisconnect:
            pass

        call_args = session_manager.gemini_client.send_text.call_args[0][0]
        self.assertIn("ОТВЕЧАЙ ИСКЛЮЧИТЕЛЬНО НА РУССКОМ ЯЗЫКЕ", call_args)

    async def test_recover_russian_enforcement(self):
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()

        session_manager.last_spoken_words = "Context."
        mock_ws.receive_text.side_effect = [
            json.dumps({"type": "recover"}),
            WebSocketDisconnect()
        ]

        try:
            await admin_websocket(mock_ws)
        except WebSocketDisconnect:
            pass

        call_args = session_manager.gemini_client.send_text.call_args[0][0]
        self.assertIn("ОТВЕЧАЙ ИСКЛЮЧИТЕЛЬНО НА РУССКОМ ЯЗЫКЕ", call_args)

if __name__ == '__main__':
    unittest.main()
