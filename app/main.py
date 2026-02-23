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
    DRAFT_MODE_SUFFIX,
    RECOVERY_PROMPT_TEMPLATE,
    SUMMARIZE_FOR_COMPRESSION_PROMPT,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Session Manager for Global State ---
class SessionManager:
    def __init__(self):
        self.active_client_ws: Optional[WebSocket] = None
        self.admin_websockets: list[WebSocket] = []
        self.gemini_client: Optional[GeminiClient] = None
        self.transcript: list[dict] = []  # [{"role": "user", "text": "Hi"}, ...]
        self.current_draft: str = ""
        self.state: dict = {
             "speaking_enabled": True,
             "speaker_active": False,
             "processing_summary": False,
             "intro_active": False,
             "draft_mode_enabled": True
        }
        self.last_spoken_words: str = ""
        self.client_mode: str = "default"

    async def broadcast_log(self, role: str, text: str):
        entry = {"role": role, "text": text}
        self.transcript.append(entry)
        if len(self.transcript) > 50:
            self.transcript.pop(0)

        msg = json.dumps({"type": "log", "role": role, "text": text})
        for ws in self.admin_websockets:
            try:
                await ws.send_text(msg)
            except:
                pass

    async def broadcast_draft(self, text: str):
        self.current_draft = text
        msg = json.dumps({"type": "draft", "text": text})
        for ws in self.admin_websockets:
            try:
                await ws.send_text(msg)
            except:
                pass

    async def broadcast_status(self, message: str):
         msg = json.dumps({"type": "status", "message": message})
         for ws in self.admin_websockets:
            try:
                await ws.send_text(msg)
            except:
                pass

    async def send_filler(self, filler_name: str):
        if self.active_client_ws:
            try:
                # Load audio
                path = f"app/static/fillers/{filler_name}.wav"
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        # Skip header? Simli usually expects PCM or WAV.
                        # Assuming client handles WAV or we strip header.
                        # If simple PCM expected, skip 44 bytes.
                        # But `send_to_client` logic usually receives raw PCM from Gemini.
                        # If we send WAV to client, frontend might need to handle it.
                        # Let's send raw bytes.
                        data = f.read()
                        # If it's a valid WAV, client might play it.
                        # For now, send as is.
                        await self.active_client_ws.send_bytes(data)
                        logger.info(f"Sent filler: {filler_name}")
            except Exception as e:
                logger.error(f"Failed to send filler: {e}")

session_manager = SessionManager()

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

@app.get("/admin")
async def get_admin(request: Request):
    return templates.TemplateResponse(request=request, name="admin.html")

@app.websocket("/ws/admin")
async def admin_websocket(websocket: WebSocket):
    await websocket.accept()
    session_manager.admin_websockets.append(websocket)
    logger.info("Admin connected")

    # Send history
    for entry in session_manager.transcript:
        await websocket.send_text(json.dumps(entry))

    if session_manager.current_draft:
        await websocket.send_text(json.dumps({"type": "draft", "text": session_manager.current_draft}))

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                command = msg.get("type")

                if command == "approve":
                    if session_manager.gemini_client:
                        logger.info("Admin approved draft.")
                        await session_manager.gemini_client.send_text("APPROVED")
                        session_manager.current_draft = ""
                        await session_manager.broadcast_status("Draft Approved")

                elif command == "reject":
                     if session_manager.gemini_client:
                        logger.info("Admin rejected draft.")
                        await session_manager.gemini_client.send_text("REJECTED. DO NOT SPEAK. Reset plan.")
                        session_manager.current_draft = ""
                        await session_manager.broadcast_status("Draft Rejected")

                elif command == "feedback":
                    text = msg.get("text")
                    if session_manager.gemini_client and text:
                        await session_manager.gemini_client.send_text(f"FEEDBACK: {text}. Rewrite plan.")
                        await session_manager.broadcast_status("Feedback Sent")

                elif command == "manual":
                    text = msg.get("text")
                    if session_manager.gemini_client and text:
                        # Direct speak override
                        await session_manager.gemini_client.send_text(f"IGNORE PLAN. SAY EXACTLY: {text}")
                        session_manager.current_draft = ""
                        await session_manager.broadcast_status("Manual Override Sent")

                elif command == "trigger_filler":
                    await session_manager.send_filler("thinking")
                    await session_manager.broadcast_status("Filler Triggered")

                elif command == "compress":
                    if session_manager.gemini_client:
                        await session_manager.gemini_client.send_text(SUMMARIZE_FOR_COMPRESSION_PROMPT)
                        await session_manager.broadcast_status("Context Compression Requested")

                elif command == "recover":
                    if session_manager.gemini_client:
                         prompt = RECOVERY_PROMPT_TEMPLATE.format(last_words=session_manager.last_spoken_words[-50:])
                         await session_manager.gemini_client.send_text(prompt)
                         await session_manager.broadcast_status("Recovery Prompt Sent")

            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        if websocket in session_manager.admin_websockets:
            session_manager.admin_websockets.remove(websocket)
        logger.info("Admin disconnected")
    except Exception as e:
        logger.error(f"Admin WS error: {e}")
        if websocket in session_manager.admin_websockets:
            session_manager.admin_websockets.remove(websocket)

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
    session_manager.active_client_ws = websocket
    session_manager.client_mode = mode

    identifier = SpeakerIdentifier(encoder=global_encoder, speakers=global_speakers)
    current_speaker = None
    loop = asyncio.get_running_loop()

    # State for Panel Mode (Synced with session manager)
    state = session_manager.state

    # Shared buffer for summary audio (buffered when passive, flushed when active)
    summary_buffer = bytearray()

    # --- Recovery Loop ---
    while True:
        try:
            gemini_client = GeminiClient()
            session_manager.gemini_client = gemini_client

            # System Instructions
            if mode == "speaker":
                system_instruction = SPEAKER_MODE_INSTRUCTION
            elif mode == "panel":
                system_instruction = PANEL_MODE_INSTRUCTION
            else:
                system_instruction = DEFAULT_INSTRUCTION

            if state.get("draft_mode_enabled"):
                system_instruction += DRAFT_MODE_SUFFIX

            await gemini_client.connect(system_instruction=system_instruction)

            async def receive_from_client():
                """Читает микрофон. Использует raw receive() для защиты от ошибок типов."""
                nonlocal current_speaker
                try:
                    while True:
                        # Используем receive() вместо receive_bytes(), чтобы проверить тип кадра
                        message = await websocket.receive()

                        if message["type"] == "websocket.disconnect":
                            raise WebSocketDisconnect

                        try:
                            if message["type"] == "websocket.receive":
                                if "bytes" in message and message["bytes"]:
                                    data = message["bytes"]

                                    # Обработка спикера
                                    segment = identifier.consume_audio(data)
                                    if segment is not None:
                                        speaker = await loop.run_in_executor(None, identifier.identify_speaker, segment)
                                        if speaker and speaker != current_speaker:
                                            current_speaker = speaker
                                            logger.info(f"Speaker changed to: {speaker}")
                                            await gemini_client.send_text(f"[User Changed: {speaker}]")
                                            await session_manager.broadcast_log("system", f"Speaker changed: {speaker}")

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
                    raise
                except Exception as e:
                    logger.error(f"Critical error receiving from client: {e}")
                    raise

            async def send_to_client():
                """Отправляет ответы клиенту."""
                audio_buffer = bytearray() # Local buffer for chunking
                MIN_CHUNK_SIZE = 4096
                is_silenced = False

                # Пауза для Simli
                logger.info("Pausing for Simli stabilization...")
                await asyncio.sleep(1.5)

                if mode == "speaker":
                     await gemini_client.send_text(SIMLI_WARMUP_SPEAKER)
                elif mode == "panel":
                     await gemini_client.send_text(SIMLI_WARMUP_PANEL)
                else:
                    await gemini_client.send_text(SIMLI_WARMUP_DEFAULT)

                try:
                    async for chunk in gemini_client.receive():
                        if isinstance(chunk, bytes):
                            is_silenced = False
                            should_send = True
                            resampled_audio = await loop.run_in_executor(None, resample_audio_sync, chunk)

                            if mode == "panel":
                                if not state["speaking_enabled"]: should_send = False
                            elif mode == "speaker":
                                if state["processing_summary"]:
                                    if resampled_audio: summary_buffer.extend(resampled_audio)
                                    should_send = False
                                elif not state["speaker_active"] and not state["intro_active"]:
                                    should_send = False

                            if is_silenced: should_send = False

                            if should_send and resampled_audio:
                                audio_buffer.extend(resampled_audio)
                                if len(audio_buffer) >= MIN_CHUNK_SIZE:
                                    await websocket.send_bytes(bytes(audio_buffer))
                                    audio_buffer.clear()

                        elif isinstance(chunk, str):
                            # --- New Logic: Draft & Logs ---
                            text_chunk = chunk.strip()
                            if text_chunk:
                                await session_manager.broadcast_log("ai", text_chunk)
                                session_manager.last_spoken_words += " " + text_chunk
                                if len(session_manager.last_spoken_words) > 500:
                                    session_manager.last_spoken_words = session_manager.last_spoken_words[-500:]

                                if "PLAN:" in text_chunk:
                                    plan_text = text_chunk
                                    await session_manager.broadcast_draft(plan_text)
                                    await session_manager.send_filler("thinking")
                                    continue

                                if "[SILENCE]" in text_chunk:
                                    if len(audio_buffer) > 0 and not is_silenced:
                                        should_flush = True
                                        if mode == "panel" and not state["speaking_enabled"]: should_flush = False
                                        elif mode == "speaker":
                                            if state["processing_summary"]: should_flush = False
                                            elif not state["speaker_active"] and not state["intro_active"]: should_flush = False

                                        if should_flush:
                                            await websocket.send_bytes(bytes(audio_buffer))

                                    is_silenced = True
                                    audio_buffer.clear()

                                    if mode == "speaker":
                                        if state["processing_summary"]:
                                            state["processing_summary"] = False
                                            await websocket.send_text(json.dumps({"type": "summary_done"}))
                                            if len(summary_buffer) > 0:
                                                await websocket.send_bytes(bytes(summary_buffer))
                                                summary_buffer.clear()
                                        if state["intro_active"]:
                                            state["intro_active"] = False
                                    logger.info("Silence token received.")
                                else:
                                    if text_chunk: is_silenced = False
                                    log_msg = {"type": "log", "role": "ai", "text": text_chunk}
                                    try:
                                        await websocket.send_text(json.dumps(log_msg))
                                    except Exception as e:
                                        logger.error(f"Failed to send text log: {e}")
                                    logger.info(f"Received text: {chunk}")

                    # Flush remaining
                    if len(audio_buffer) > 0 and not is_silenced:
                        should_send = True
                        if mode == "panel" and not state["speaking_enabled"]: should_send = False
                        elif mode == "speaker":
                             if state["processing_summary"]: should_send = False
                             elif not state["speaker_active"] and not state["intro_active"]: should_send = False

                        if should_send:
                            await websocket.send_bytes(bytes(audio_buffer))

                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    raise

            async def keep_alive():
                """Фоновая задача."""
                try:
                    while True:
                        await asyncio.sleep(5)
                        await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    pass

            tasks = [
                asyncio.create_task(receive_from_client()),
                asyncio.create_task(send_to_client()),
                asyncio.create_task(keep_alive())
            ]

            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending: task.cancel()
                for task in done:
                    if task.exception(): raise task.exception()

            except WebSocketDisconnect:
                logger.info("Client disconnected.")
                break # Exit outer loop
            except Exception as e:
                logger.error(f"Gemini/Internal Error: {e}. Recovering...")
                await gemini_client.close()
                await session_manager.broadcast_status("Recovering Connection...")
                await asyncio.sleep(1)

        finally:
            if 'gemini_client' in locals() and gemini_client:
                # Ensure closed if we broke out of loop
                pass

    session_manager.active_client_ws = None
    try:
        await websocket.close()
    except:
        pass

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
