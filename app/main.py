from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import os
import asyncio
import logging
import re
from dotenv import load_dotenv
import numpy as np
import torch
import torchaudio
from openai import AsyncOpenAI

from app.gemini import GeminiClient
from app.speaker_id import SpeakerIdentifier
from resemblyzer import VoiceEncoder

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

global_encoder = None
global_speakers = {}
global_resampler = None

# Initialize OpenAI Client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.on_event("startup")
async def startup_event():
    global global_encoder, global_speakers, global_resampler
    try:
        global_encoder = VoiceEncoder()
        logger.info("Global VoiceEncoder loaded")

        # Init Resampler (24kHz -> 16kHz)
        # We initialize it here to avoid re-creation overhead on every chunk
        global_resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)
        logger.info("Global Resampler loaded")

        # Load speakers
        temp_identifier = SpeakerIdentifier(encoder=global_encoder)
        temp_identifier.load_speakers_from_folder("app/static/samples")
        global_speakers = temp_identifier.speakers
        logger.info(f"Loaded {len(global_speakers)} speakers")

    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/simli/config")
def get_simli_config():
    return {
        "apiKey": os.getenv("SIMLI_API_KEY"),
        "faceID": os.getenv("SIMLI_FACE_ID"),
    }

def resample_audio_sync(audio_bytes: bytes) -> bytes:
    """
    Synchronous function to be run in executor.
    """
    if not audio_bytes or not global_resampler:
        return b""
    try:
        # Convert raw bytes (int16) to float32 tensor
        waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        waveform = torch.from_numpy(waveform).unsqueeze(0)  # Shape (1, L)

        # Resample using the global pre-initialized resampler
        resampled_waveform = global_resampler(waveform)

        # Convert back to int16 bytes
        resampled_np = resampled_waveform.squeeze(0).numpy().astype(np.int16)
        return resampled_np.tobytes()
    except Exception as e:
        logger.error(f"Resampling error: {e}")
        return audio_bytes # Fallback

async def text_to_speech_pcm(text: str) -> bytes:
    try:
        # OpenAI TTS-1 output is 24kHz for 'pcm' format by default
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="pcm"
        )
        audio_data = response.content

        # Run CPU-bound resampling in a thread pool to avoid blocking async loop
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, resample_audio_sync, audio_data)

    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return b""

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    gemini_client = GeminiClient()
    identifier = SpeakerIdentifier(encoder=global_encoder, speakers=global_speakers)
    current_speaker = None
    loop = asyncio.get_running_loop()

    try:
        # Dos System Instruction
        system_instruction = (
            "You are Dos, an expert and friendly AI assistant. "
            "You are listening to a conversation of a group of up to 8 people. "
            "You must remain absolutely silent unless the input starts explicitly with the name 'Dos'. "
            "Ignore general commands like 'Answer me', 'Are you listening?' if the name 'Dos' is missing. "
            "Allowed patterns: 'Dos, [question]', 'Dos, answer', 'Dos, your opinion'. "
            "If asked 'Who are you?' or 'Introduce yourself', reply exactly: 'Я ИИ спикер Dos'. "
            "When you answer, address the interlocutor by name if you can determine it (e.g., 'Yes, Tamerlan...'). "
            "Be brief, professional, but with a warm tone."
        )
        await gemini_client.connect(system_instruction=system_instruction)

        # Trigger Welcome Message
        # The assistant must initiate the dialogue.
        await gemini_client.send_text('Please say exactly: "Я ИИ спикер Dos. Сегодня я буду вместе с вами разбирать и участвовать в теме обсуждения, которую вы зададите."')

        async def receive_from_client():
            nonlocal current_speaker
            try:
                while True:
                    # Expecting raw PCM bytes from user microphone
                    data = await websocket.receive_bytes()

                    # Speaker ID (Run in executor to avoid blocking)
                    speaker = await loop.run_in_executor(None, identifier.process_chunk, data)

                    if speaker and speaker != current_speaker:
                        current_speaker = speaker
                        logger.info(f"Speaker changed to: {speaker}")
                        await gemini_client.send_text(f"[User Changed: {speaker}]")

                    await gemini_client.send_audio(data)
            except WebSocketDisconnect:
                logger.info("Client disconnected")
            except Exception as e:
                logger.error(f"Error receiving from client: {e}")

        async def send_to_client():
            text_buffer = ""
            try:
                async for chunk in gemini_client.receive():
                    if isinstance(chunk, str):
                        text_buffer += chunk

                        # Buffer until we have a complete sentence to ensure high-quality TTS
                        # Split by punctuation followed by whitespace
                        sentences = re.split(r'(?<=[.!?])\s+', text_buffer)

                        if len(sentences) > 1:
                            # We have at least one complete sentence
                            to_process = sentences[:-1]
                            text_buffer = sentences[-1]

                            for sentence in to_process:
                                if sentence.strip():
                                    logger.info(f"TTS Sentence: {sentence[:30]}...")
                                    audio = await text_to_speech_pcm(sentence)
                                    if audio: await websocket.send_bytes(audio)

                        # Fallback: if buffer gets too large without punctuation, flush it
                        if len(text_buffer) > 250:
                            logger.info("Flushing large text buffer")
                            audio = await text_to_speech_pcm(text_buffer)
                            text_buffer = ""
                            if audio: await websocket.send_bytes(audio)

                    elif isinstance(chunk, bytes):
                        # Audio passthrough
                        await websocket.send_bytes(chunk)

                # Flush remaining text at end of stream
                if text_buffer.strip():
                    audio = await text_to_speech_pcm(text_buffer)
                    if audio: await websocket.send_bytes(audio)

            except Exception as e:
                logger.error(f"Error sending to client: {e}")

        await asyncio.gather(receive_from_client(), send_to_client())

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await gemini_client.close()
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
