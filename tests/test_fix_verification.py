# ... (imports same as before)
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies BEFORE importing app
sys.modules['resemblyzer'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['torchaudio.transforms'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['soundfile'] = MagicMock()

try:
    import websockets
except ImportError:
    sys.modules['websockets'] = MagicMock()

class MockGeminiClient:
    def __init__(self):
        self.ws = MagicMock()

    async def connect(self, system_instruction=None):
        pass

    async def send_text(self, text):
        print(f"MockGemini: Received Text -> {text}")

    async def send_audio(self, audio):
        pass

    async def receive(self):
        # Simulate sufficient audio to trigger MIN_CHUNK_SIZE
        # MIN_CHUNK_SIZE = 4096
        yield b"A" * 5000
        yield "[SILENCE]"
        await asyncio.sleep(0.1)

    async def close(self):
        pass

import app.main
app.main.GeminiClient = MockGeminiClient
# PATCH resample_audio_sync to just return input bytes
app.main.resample_audio_sync = lambda x: x

fastapi_app = app.main.app
fastapi_app.router.startup_handlers = []

from fastapi.testclient import TestClient
client = TestClient(fastapi_app)

def verify_fix():
    print("\n--- Verifying Passive Command Execution ---")
    with client.websocket_connect("/ws?mode=speaker") as websocket:
        print("Connected in Passive Mode (Default).")

        # 1. Trigger Introduce WITHOUT toggling active
        print("Sending trigger_introduce...")
        websocket.send_json({"type": "trigger_introduce"})

        # We expect to receive audio bytes.
        try:
            data = websocket.receive_bytes()
            if len(data) > 0:
                print(f"SUCCESS: Received {len(data)} bytes of audio in Passive Mode.")
            else:
                print(f"WARNING: Received empty bytes: {data}")
        except Exception as e:
            print(f"FAIL: Did not receive expected audio bytes: {e}")
            try:
                text = websocket.receive_text()
                print(f"Received text instead: {text}")
            except:
                pass

if __name__ == "__main__":
    try:
        verify_fix()
    except Exception as e:
        print(f"Test Failed: {e}")
