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
from app.prompts import (
    SPEAKER_MODE_INSTRUCTION,
    PANEL_MODE_INSTRUCTION,
    DEFAULT_INSTRUCTION,
    PASSIVE_MODE_COMMAND,
    INTRODUCE_COMMAND,
    SUMMARIZE_COMMAND_SPEAKER,
    SUMMARIZE_PROMPT_PANEL,
    SIMLI_WARMUP_SPEAKER,
    SIMLI_WARMUP_PANEL,
    SIMLI_WARMUP_DEFAULT,
)

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
@app.get("/panel")
@app.get("/speaker")
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/simli/config")
def get_simli_config():
    return {
        "apiKey": os.getenv("SIMLI_API_KEY"),
        "faceID": os.getenv("SIMLI_FACE_ID"),
    }

def resample_audio_sync(audio_bytes: bytes) -> bytes:
    if not audio_bytes:
        return b""
    if not global_resampler:
        return audio_bytes
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
async def websocket_endpoint(websocket: WebSocket, mode: str = "default"):
    await websocket.accept()

    gemini_client = GeminiClient()
    identifier = SpeakerIdentifier(encoder=global_encoder, speakers=global_speakers)
    current_speaker = None
    loop = asyncio.get_running_loop()

    # State for Panel Mode
    state = {
        "speaking_enabled": True,
        "speaker_active": False,  # Default to Passive for Speaker Mode
        "processing_summary": False,
        "intro_active": False
    }

    # Shared buffer for summary audio (buffered when passive, flushed when active)
    summary_buffer = bytearray()

    try:
        # Determine System Instruction based on Mode

        # System Instructions
        if mode == "speaker":
            system_instruction = SPEAKER_MODE_INSTRUCTION
        elif mode == "panel":
            system_instruction = PANEL_MODE_INSTRUCTION
        else:
            system_instruction = DEFAULT_INSTRUCTION
        
        await gemini_client.connect(system_instruction=system_instruction)

        async def receive_from_client():
            """Читает микрофон. Использует raw receive() для защиты от ошибок типов."""
            nonlocal current_speaker
            try:
                while True:
                    # Используем receive() вместо receive_bytes(), чтобы проверить тип кадра
                    message = await websocket.receive()
                    
                    if message["type"] == "websocket.disconnect":
                        logger.info("Client disconnected (Event)")
                        break
                    
                    try:
                        if message["type"] == "websocket.receive":
                            if "bytes" in message and message["bytes"]:
                                data = message["bytes"]

                                # Обработка спикера
                                speaker = await loop.run_in_executor(None, identifier.process_chunk, data)
                                if speaker and speaker != current_speaker:
                                    current_speaker = speaker
                                    logger.info(f"Speaker changed to: {speaker}")
                                    await gemini_client.send_text(f"[User Changed: {speaker}]")

                                # Отправка аудио в Gemini (Context Preservation: Always send audio regardless of Mute state - Verified)
                                await gemini_client.send_audio(data)
                            
                            elif "text" in message:
                                # Обработка JSON команд от клиента
                                try:
                                    msg_data = json.loads(message["text"])

                                    if not isinstance(msg_data, dict):
                                        continue

                                    if msg_data.get("type") == "mute_toggle":
                                        state["speaking_enabled"] = msg_data.get("enabled", True)
                                        logger.info(f"Mute toggle: speaking_enabled={state['speaking_enabled']}")

                                    elif msg_data.get("type") == "toggle_active":
                                        if mode == "speaker":
                                            enabled = msg_data.get("enabled", False)
                                            state["speaker_active"] = enabled

                                            if not enabled:
                                                await gemini_client.send_text(PASSIVE_MODE_COMMAND)
                                            else:
                                                # If enabling active mode, flush buffer if any
                                                if len(summary_buffer) > 0:
                                                    logger.info(f"Flushing summary buffer: {len(summary_buffer)} bytes")
                                                    await websocket.send_bytes(bytes(summary_buffer))
                                                    summary_buffer.clear()

                                    elif msg_data.get("type") == "trigger_introduce":
                                        if mode == "speaker":
                                            state["intro_active"] = True
                                            await gemini_client.send_text(INTRODUCE_COMMAND)

                                    elif msg_data.get("type") == "trigger_summary":
                                        logger.info("Triggering summary generation...")
                                        if mode == "speaker":
                                            state["processing_summary"] = True
                                            await gemini_client.send_text(SUMMARIZE_COMMAND_SPEAKER)
                                        else:
                                            await gemini_client.send_text(SUMMARIZE_PROMPT_PANEL)
                                except json.JSONDecodeError:
                                    pass
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

            except WebSocketDisconnect:
                logger.info("Client disconnected (Exception)")
            except Exception as e:
                logger.error(f"Critical error receiving from client: {e}")

        async def send_to_client():
            """Отправляет ответы клиенту."""
            loop = asyncio.get_running_loop()
            audio_buffer = bytearray() # Local buffer for chunking
            MIN_CHUNK_SIZE = 4096
            is_silenced = False

            # Пауза для Simli
            logger.info("Pausing for Simli stabilization...")
            await asyncio.sleep(1.5)

            # Приветствие (зависит от режима)
            if mode == "speaker":
                 await gemini_client.send_text(SIMLI_WARMUP_SPEAKER)
            elif mode == "panel":
                 await gemini_client.send_text(SIMLI_WARMUP_PANEL)
            else:
                await gemini_client.send_text(SIMLI_WARMUP_DEFAULT)

            try:
                # Стандартный цикл чтения (без прерываний)
                async for chunk in gemini_client.receive():
                    if isinstance(chunk, bytes):
                        # Авто-сброс флага тишины при получении аудио (новая фраза или ответ)
                        is_silenced = False
                        should_send = True
                        resampled_audio = await loop.run_in_executor(None, resample_audio_sync, chunk)

                        # Логика Output Gating
                        if mode == "panel":
                            if not state["speaking_enabled"]:
                                should_send = False

                        elif mode == "speaker":
                            if state["processing_summary"]:
                                # Accumulate audio in buffer
                                if resampled_audio:
                                    summary_buffer.extend(resampled_audio)
                                should_send = False
                            elif not state["speaker_active"] and not state["intro_active"]:
                                should_send = False

                        if is_silenced:
                            should_send = False

                        if should_send and resampled_audio:
                            audio_buffer.extend(resampled_audio)
                            if len(audio_buffer) >= MIN_CHUNK_SIZE:
                                await websocket.send_bytes(bytes(audio_buffer))
                                audio_buffer.clear()

                    elif isinstance(chunk, str):
                        if "[SILENCE]" in chunk:
                            is_silenced = True
                            audio_buffer.clear() # Очищаем текущий буфер вывода

                            if mode == "speaker":
                                # Генерация саммари завершена
                                if state["processing_summary"]:
                                    state["processing_summary"] = False
                                    await websocket.send_text(json.dumps({"type": "summary_done"}))

                                    # Edge case: If active mode was enabled during generation, send buffer now
                                    if state["speaker_active"] and len(summary_buffer) > 0:
                                        await websocket.send_bytes(bytes(summary_buffer))
                                        summary_buffer.clear()

                                # Сброс флага Intro
                                if state["intro_active"]:
                                    state["intro_active"] = False

                            logger.info("Silence token received.")
                        else:
                            # Text Fallback: Log non-silence text
                            if chunk.strip(): is_silenced = False
                            log_msg = {"type": "log", "role": "ai", "text": chunk}
                            try:
                                await websocket.send_text(json.dumps(log_msg))
                            except Exception as e:
                                logger.error(f"Failed to send text log: {e}")
                            logger.info(f"Received text: {chunk}")

                # Отправляем остатки аудио
                if len(audio_buffer) > 0 and not is_silenced:
                    should_send = True
                    if mode == "panel" and not state["speaking_enabled"]:
                        should_send = False
                    elif mode == "speaker":
                         if state["processing_summary"]:
                             should_send = False # Already buffered
                         elif not state["speaker_active"] and not state["intro_active"]:
                            should_send = False

                    if should_send:
                        await websocket.send_bytes(bytes(audio_buffer))

            except Exception as e:
                logger.error(f"Error sending to client: {e}")

        async def keep_alive():
            """Фоновая задача: отправляет пинг каждые 5 секунд, чтобы канал не умер."""
            try:
                while True:
                    await asyncio.sleep(5)
                    # Шлем пинг, чтобы облако/браузер не разорвали соединение (Error 1011)
                    await websocket.send_text(json.dumps({"type": "ping"}))
            except Exception:
                pass

        # Запускаем ВСЕ задачи параллельно: чтение микрофона, отправку звука и пинги
        await asyncio.gather(receive_from_client(), send_to_client(), keep_alive())

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
    uvicorn.run(app, host="0.0.0.0", port=port)
