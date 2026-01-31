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
import json
from dotenv import load_dotenv
import numpy as np
import torch
import torchaudio

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

@app.on_event("startup")
async def startup_event():
    global global_encoder, global_speakers, global_resampler
    try:
        global_encoder = VoiceEncoder()
        logger.info("Global VoiceEncoder loaded")
        global_resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)
        logger.info("Global Resampler loaded")
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
    if not audio_bytes or not global_resampler:
        return b""
    try:
        waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        resampled_waveform = global_resampler(waveform)
        resampled_np = resampled_waveform.squeeze(0).numpy().astype(np.int16)
        return resampled_np.tobytes()
    except Exception as e:
        logger.error(f"Resampling error: {e}")
        return audio_bytes

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    gemini_client = GeminiClient()
    identifier = SpeakerIdentifier(encoder=global_encoder, speakers=global_speakers)
    current_speaker = None
    loop = asyncio.get_running_loop()

    try:
        # System Instruction
        system_instruction = (
            "You are Dos, an expert AI assistant and passive analyst. "
            "ROLE: Deep Dive Expert. Engage in deep analytical discussion. "
            "MODE: AUDIO-ONLY. "
            "CRITICAL RULE: NEVER output text thoughts, internal monologue, or explanations in the audio stream. "
            "ACTIVATION: You are listening to a conversation. "
            "IF the user's input explicitly starts with or contains the name 'Dos' (or 'Дос'): "
            "  - Generate a comprehensive, structured, and expert-level audio response in Russian. "
            "  - Use the full context of the conversation. "
            "IF the name 'Dos' is NOT heard: "
            "  - Output EXACTLY the text token: [SILENCE] "
            "  - Do NOT generate any audio. "
            "IDENTITY: 'Я ИИ спикер Dos'. "
            "LANGUAGE: Russian. "
            "Speak clearly, with a moderate pace, articulating words distinctively to ensure good lip-sync."
        )
        
        # 1. Сначала подключаемся к Gemini
        await gemini_client.connect(system_instruction=system_instruction)

        # 2. Определяем задачи, но не запускаем блокирующий sleep в главном потоке!
        
        async def receive_from_client():
            """Читает микрофон постоянно, чтобы не забивать буфер"""
            nonlocal current_speaker
            try:
                while True:
                    data = await websocket.receive_bytes()
                    # Отправляем в Gemini (он будет буферизировать, пока мы не слушаем ответ)
                    speaker = await loop.run_in_executor(None, identifier.process_chunk, data)
                    if speaker and speaker != current_speaker:
                        current_speaker = speaker
                        logger.info(f"Speaker changed to: {speaker}")
                        await gemini_client.send_text(f"[User Changed: {speaker}]")
                    
                    await gemini_client.send_audio(data)
            except WebSocketDisconnect:
                logger.info("Client disconnected (Receive Loop)")
            except Exception as e:
                logger.error(f"Error receiving from client: {e}")

        async def send_to_client():
            """Отвечает за Simli и отправку ответов"""
            loop = asyncio.get_running_loop()
            audio_buffer = bytearray()
            MIN_CHUNK_SIZE = 4096
            is_silenced = False

            # ПАУЗА ЗДЕСЬ - внутри асинхронной задачи, не блокируя чтение!
            logger.info("Pausing for Simli stabilization...")
            await asyncio.sleep(1.5)

            # Приветственное сообщение
            await gemini_client.send_text(
                'Generate audio immediately. Say exactly this phrase with energy: '
                '"Я ИИ спикер Dos. Сегодня я буду вместе с вами разбирать и участвовать в теме обсуждения, которую вы зададите."'
            )

            try:
                async for chunk in gemini_client.receive():
                    if isinstance(chunk, bytes):
                        if is_silenced: continue
                        resampled_audio = await loop.run_in_executor(None, resample_audio_sync, chunk)
                        if resampled_audio:
                            audio_buffer.extend(resampled_audio)
                            if len(audio_buffer) >= MIN_CHUNK_SIZE:
                                await websocket.send_bytes(bytes(audio_buffer))
                                audio_buffer.clear()

                    elif isinstance(chunk, str):
                        if "[SILENCE]" in chunk:
                            is_silenced = True
                            audio_buffer.clear()
                            logger.info("Silence token received.")
                        else:
                            if chunk.strip(): is_silenced = False
                            # Отправляем JSON (ASCII-safe)
                            log_msg = {"type": "log", "role": "ai", "text": chunk}
                            await websocket.send_text(json.dumps(log_msg)) # ensure_ascii=True по умолчанию
                            logger.info(f"Received text: {chunk}")

                if len(audio_buffer) > 0 and not is_silenced:
                    await websocket.send_bytes(bytes(audio_buffer))

            except Exception as e:
                logger.error(f"Error sending to client: {e}")

        # 3. Запускаем обе задачи параллельно
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
    port = int(os.getenv("PORT", 8080))
    # Используем wsproto (убедитесь, что он добавлен в requirements.txt)
    uvicorn.run(app, host="0.0.0.0", port=port, ws="wsproto")
