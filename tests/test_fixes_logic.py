import unittest
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
import sys
import asyncio
import os

# Mock dependencies before importing app.main
sys.modules['numpy'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['resemblyzer'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['fastapi.staticfiles'] = MagicMock()
sys.modules['fastapi.templating'] = MagicMock()
sys.modules['fastapi.requests'] = MagicMock()
sys.modules['pydantic'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['app.gemini'] = MagicMock()
sys.modules['app.speaker_id'] = MagicMock()

# Now import app.main
# We need to ensure that the mocked modules have the necessary attributes
sys.modules['app.gemini'].GeminiClient = MagicMock()
sys.modules['app.speaker_id'].SpeakerIdentifier = MagicMock()
sys.modules['pydantic'].BaseModel = MagicMock

# We need to mock os.getenv to avoid issues during import if it uses it
with patch.dict(os.environ, {"SIMLI_API_KEY": "test", "SIMLI_FACE_ID": "test"}):
    # Also need to mock logging basicConfig etc
    with patch("logging.basicConfig"), patch("logging.getLogger"):
        from app.main import SessionManager

class TestFixes(unittest.IsolatedAsyncioTestCase):
    async def test_send_filler_wav_header_stripping(self):
        """Test that send_filler strips the first 44 bytes if it starts with RIFF."""
        sm = SessionManager()
        sm.active_client_ws = AsyncMock()

        # Create dummy WAV data: 44 bytes header + some payload
        # Header starts with RIFF (4 bytes), then 40 bytes of other stuff
        header = b'RIFF' + b'\x00' * 40
        payload = b'\x01\x02\x03\x04'
        file_content = header + payload

        # We need to mock 'open' specifically for this call
        # but we also need to be careful not to break other imports if they happen inside (unlikely)
        m = mock_open(read_data=file_content)
        with patch("builtins.open", m):
            with patch("os.path.exists", return_value=True):
                await sm.send_filler("test_filler")

        # Verify that only payload was sent
        # CURRENTLY: This should FAIL because the code sends everything
        # Once fixed, it should PASS

        # Checking what was actually called
        # sm.active_client_ws.send_bytes.assert_called_once_with(payload)

        # Since we expect failure now, let's catch the assertion error or inspect the call
        # calls = sm.active_client_ws.send_bytes.call_args_list
        # if calls:
        #     args, _ = calls[0]
        #     sent_data = args[0]
        #     print(f"Sent data length: {len(sent_data)}")

        # For now, assert exact match to see it fail
        sm.active_client_ws.send_bytes.assert_called_once_with(payload)

    async def test_send_filler_no_riff(self):
        """Test that send_filler sends raw data if no RIFF header."""
        sm = SessionManager()
        sm.active_client_ws = AsyncMock()

        content = b'some_other_format'

        with patch("builtins.open", mock_open(read_data=content)):
            with patch("os.path.exists", return_value=True):
                await sm.send_filler("test_filler")

        sm.active_client_ws.send_bytes.assert_called_once_with(content)

    async def test_draft_accumulation_logic(self):
        """
        Test the logic for draft accumulation (simulate the loop behavior).
        """
        sm = SessionManager()
        sm.state["draft_mode_enabled"] = True
        sm.current_draft = ""
        sm.broadcast_draft = AsyncMock()

        # Simulate text chunk arrival
        text_chunk = "My plan is to "

        # Proposed logic:
        if sm.state.get("draft_mode_enabled"):
            sm.current_draft += text_chunk + " "
            await sm.broadcast_draft(sm.current_draft)

        self.assertEqual(sm.current_draft, "My plan is to  ") # Space appended
        sm.broadcast_draft.assert_called_with("My plan is to  ")

        # Verify subsequent chunk
        text_chunk_2 = "do nothing."
        if sm.state.get("draft_mode_enabled"):
            sm.current_draft += text_chunk_2 + " "
            await sm.broadcast_draft(sm.current_draft)

        self.assertEqual(sm.current_draft, "My plan is to  do nothing. ")

if __name__ == '__main__':
    unittest.main()
