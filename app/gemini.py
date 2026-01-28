import websockets
import json
import asyncio
import os
import base64
import logging

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        # Using v1alpha as per documentation for experimental models
        self.url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        self.ws = None

    async def connect(self, system_instruction: str = None):
        """Establishes the WebSocket connection and sends setup message."""
        try:
            self.ws = await websockets.connect(self.url)
            await self._send_setup(system_instruction)
            logger.info("Connected to Gemini Live API")
        except Exception as e:
            logger.error(f"Failed to connect to Gemini: {e}")
            raise

    async def _send_setup(self, system_instruction: str = None):
        setup_msg = {
            "setup": {
                "model": "models/gemini-2.5-flash-native-audio-latest",
                "generationConfig": {
                    "responseModalities": ["TEXT"]
                }
            }
        }
        if system_instruction:
            setup_msg["setup"]["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        await self.ws.send(json.dumps(setup_msg))
        # Wait for potential setup complete or just proceed.
        # The API doesn't always send a specific "Setup Complete" message immediately,
        # but the first message usually confirms it or errors.
        # For now, we assume success if no error on send.

    async def send_audio(self, audio_data: bytes):
        """Sends raw PCM 16kHz audio data to Gemini."""
        if not self.ws:
            return

        b64_audio = base64.b64encode(audio_data).decode("utf-8")
        msg = {
            "realtimeInput": {
                "mediaChunks": [
                    {
                        "mimeType": "audio/pcm;rate=16000",
                        "data": b64_audio
                    }
                ]
            }
        }
        await self.ws.send(json.dumps(msg))

    async def send_text(self, text: str):
        """Sends text input to Gemini (e.g. context updates)."""
        if not self.ws:
            return

        msg = {
            "clientContent": {
                "turns": [
                    {
                        "role": "user",
                        "parts": [{"text": text}]
                    }
                ],
                "turnComplete": True
            }
        }
        await self.ws.send(json.dumps(msg))

    async def receive(self):
        """Yields text chunks from Gemini."""
        if not self.ws:
            return

        async for message in self.ws:
            try:
                data = json.loads(message)
                # Parse serverContent
                if "serverContent" in data:
                    server_content = data["serverContent"]
                    if "modelTurn" in server_content:
                        parts = server_content["modelTurn"].get("parts", [])
                        for part in parts:
                            if "inlineData" in part:
                                # Audio data (Not expected in TEXT mode, but just in case)
                                b64_data = part["inlineData"]["data"]
                                yield base64.b64decode(b64_data)
                            if "text" in part:
                                text = part["text"]
                                logger.info(f"Gemini Text: {text}")
                                yield text

                if "toolCall" in data:
                    logger.info("Tool call received (not implemented)")
            except Exception as e:
                logger.error(f"Error parsing message: {e}")
                continue

    async def close(self):
        if self.ws:
            await self.ws.close()
