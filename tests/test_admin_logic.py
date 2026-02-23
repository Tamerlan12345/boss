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

class TestAdminLogic(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        session_manager.admin_websockets = []
        session_manager.active_client_ws = None
        session_manager.gemini_client = None
        session_manager.current_draft = ""
        session_manager.transcript = []

    async def test_session_manager_broadcast(self):
        mock_ws = MagicMock()
        mock_ws.send_text = AsyncMock()
        session_manager.admin_websockets.append(mock_ws)
        await session_manager.broadcast_log("user", "Hello")
        self.assertTrue(mock_ws.send_text.called)

    @patch("app.main.os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=b"audiobytes")
    async def test_session_manager_send_filler(self, mock_open, mock_exists):
        mock_client_ws = MagicMock()
        mock_client_ws.send_bytes = AsyncMock()
        session_manager.active_client_ws = mock_client_ws
        await session_manager.send_filler("thinking")
        mock_client_ws.send_bytes.assert_called_with(b"audiobytes")

    async def test_admin_websocket_approve(self):
        mock_gemini = MagicMock()
        mock_gemini.send_text = AsyncMock()
        session_manager.gemini_client = mock_gemini
        session_manager.current_draft = "Plan A"

        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()
        mock_ws.receive_text.side_effect = [
            json.dumps({"type": "approve"}),
            WebSocketDisconnect()
        ]

        try:
            await admin_websocket(mock_ws)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Caught exception in test: {e}")

        mock_gemini.send_text.assert_called_with("APPROVED")

    async def test_admin_websocket_feedback(self):
        mock_gemini = MagicMock()
        mock_gemini.send_text = AsyncMock()
        session_manager.gemini_client = mock_gemini

        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()
        mock_ws.receive_text.side_effect = [
            json.dumps({"type": "feedback", "text": "No release"}),
            WebSocketDisconnect()
        ]

        try:
            await admin_websocket(mock_ws)
        except Exception:
            pass

        mock_gemini.send_text.assert_called_with("FEEDBACK: No release. Rewrite plan.")

    async def test_admin_websocket_trigger_filler(self):
        mock_client_ws = MagicMock()
        mock_client_ws.send_bytes = AsyncMock()
        session_manager.active_client_ws = mock_client_ws

        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()
        mock_ws.receive_text.side_effect = [
            json.dumps({"type": "trigger_filler"}),
            WebSocketDisconnect()
        ]

        # Patching inside the test method needs to ensure it covers the execution
        with patch("app.main.os.path.exists", return_value=True):
            with patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=b"filler"):
                try:
                    await admin_websocket(mock_ws)
                except Exception:
                    pass

        mock_client_ws.send_bytes.assert_called_with(b"filler")

    async def test_admin_websocket_recover(self):
        mock_gemini = MagicMock()
        mock_gemini.send_text = AsyncMock()
        session_manager.gemini_client = mock_gemini
        session_manager.last_spoken_words = "last words spoken"

        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()
        mock_ws.receive_text.side_effect = [
            json.dumps({"type": "recover"}),
            WebSocketDisconnect()
        ]

        try:
            await admin_websocket(mock_ws)
        except Exception:
            pass

        self.assertTrue(mock_gemini.send_text.called)
        args = mock_gemini.send_text.call_args[0][0]
        self.assertIn("last words spoken", args)
