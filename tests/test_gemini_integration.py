import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import json
import os
import asyncio
from fastapi.testclient import TestClient

# Mock heavy libraries to avoid import errors or slow loads
# We need to do this before importing app.main
with patch("resemblyzer.VoiceEncoder"), \
     patch("app.speaker_id.SpeakerIdentifier"):
    from app.main import app
    from app.gemini import GeminiClient

class TestGeminiLogic(unittest.IsolatedAsyncioTestCase):

    async def test_gemini_setup_config(self):
        """Verify GeminiClient sends correct setup configuration."""
        with patch("app.gemini.websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_connect.return_value = mock_ws

            # Ensure API Key is set
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
                client = GeminiClient()
                await client.connect(system_instruction="sys_instr")

            # Get the argument passed to ws.send
            # The first call to send should be the setup message
            args, _ = mock_ws.send.call_args
            sent_json = json.loads(args[0])

            # Assertions
            setup = sent_json.get("setup", {})
            self.assertEqual(setup.get("model"), "models/gemini-2.5-flash-native-audio-latest")

            gen_config = setup.get("generationConfig", {})
            self.assertEqual(gen_config.get("responseModalities"), ["AUDIO"])

            speech_config = gen_config.get("speechConfig", {})
            voice_name = speech_config.get("voiceConfig", {}).get("prebuiltVoiceConfig", {}).get("voiceName")
            self.assertEqual(voice_name, "Puck")

    def test_main_loop_filtering(self):
        """
        Verify that app.main.send_to_client:
        1. Ignores str (text) chunks.
        2. Resamples and sends bytes (audio) chunks.
        """
        # We simulate the GeminiClient instance behavior
        mock_gemini_instance = MagicMock()
        mock_gemini_instance.connect = AsyncMock()
        mock_gemini_instance.send_text = AsyncMock()
        mock_gemini_instance.send_audio = AsyncMock()
        mock_gemini_instance.close = AsyncMock()

        # Create an async generator for receive()
        async def mock_receive_gen():
            yield "ignore this text"
            yield b"audio_raw"
            # We pause here to let the test client consume the message
            # then we can stop or yield more.
            # Just returning will end the loop.

        mock_gemini_instance.receive = mock_receive_gen

        # Mock resample_audio_sync to just modify the bytes so we know it ran
        def mock_resample(data):
            return data + b"_resampled"

        with patch("app.main.GeminiClient", return_value=mock_gemini_instance), \
             patch("app.main.resample_audio_sync", side_effect=mock_resample) as mock_resampler, \
             patch("app.main.SpeakerIdentifier"), \
             patch("app.main.VoiceEncoder"):

            client = TestClient(app)
            with client.websocket_connect("/ws") as websocket:
                # The endpoint calls gemini_client.connect, send_text(welcome), then starts loops.
                # send_to_client loop will iterate mock_receive_gen()

                # We expect to receive ONE binary message: b"audio_raw_resampled"
                # The text "ignore this text" should produce NO output.

                data = websocket.receive_bytes()
                self.assertEqual(data, b"audio_raw_resampled")

                # If the text was sent, the TestClient would have likely raised an error
                # (since we called receive_bytes, and if it got text it complains)
                # OR it would be queued.

                # To be absolutely sure, we assert that mock_resampler was called exactly once.
                # If text was processed as audio, it would be called twice (or fail).
                self.assertEqual(mock_resampler.call_count, 1)
                mock_resampler.assert_called_with(b"audio_raw")

if __name__ == "__main__":
    unittest.main()
